//! Multimodal tensor transport resolution.
//!
//! Decides how large multimodal tensors travel from the gateway to the worker:
//! the SHM-vs-inline transport mode, its size threshold, the encoder-input wire
//! dtype, and the `/dev/shm` namespace verification that makes the SHM path safe.
//!
//! Resolution precedence for the transport mode and SHM threshold:
//! per-worker `WorkerSpec` override → router config (seeded once at startup via
//! [`init_mm_runtime_config`]) → built-in default (`inline`, 64 KiB). Product
//! environment variables are deliberately not part of runtime configuration.

use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use llm_multimodal::Modality;
use openai_protocol::worker::TransportMode;
use smg_mm_rdma::{RdmaConfig, RdmaExporter};
use tracing::{error, info, warn};

use crate::{
    config::{
        RouterConfig, DEFAULT_MULTIMODAL_RDMA_LISTEN_PORT, DEFAULT_MULTIMODAL_RDMA_POOL_SLOTS,
        DEFAULT_MULTIMODAL_RDMA_SLOT_BYTES, DEFAULT_MULTIMODAL_RDMA_WORKER_LANDING_WAIT_SECS,
        DEFAULT_MULTIMODAL_RDMA_WORKER_READ_TIMEOUT_SECS, DEFAULT_MULTIMODAL_SHM_MIN_BYTES,
        MAX_MULTIMODAL_RDMA_ARENA_BYTES,
    },
    routers::grpc::{context::WorkerSelection, proto_wrapper::mm_shm_dev_writable},
};

#[derive(Debug, Clone)]
struct MmRdmaRuntimeConfig {
    listen_ip: Option<String>,
    listen_port: u16,
    pool_slots: usize,
    slot_bytes: usize,
    worker_landing_wait_secs: u64,
    worker_read_timeout_secs: u64,
    slot_ttl_secs: Option<u64>,
}

impl Default for MmRdmaRuntimeConfig {
    fn default() -> Self {
        Self {
            listen_ip: None,
            listen_port: DEFAULT_MULTIMODAL_RDMA_LISTEN_PORT,
            pool_slots: DEFAULT_MULTIMODAL_RDMA_POOL_SLOTS,
            slot_bytes: DEFAULT_MULTIMODAL_RDMA_SLOT_BYTES,
            worker_landing_wait_secs: DEFAULT_MULTIMODAL_RDMA_WORKER_LANDING_WAIT_SECS,
            worker_read_timeout_secs: DEFAULT_MULTIMODAL_RDMA_WORKER_READ_TIMEOUT_SECS,
            slot_ttl_secs: None,
        }
    }
}

/// Process-wide multimodal runtime policy projected from the validated
/// [`RouterConfig`] before requests are served.
#[derive(Debug, Clone)]
struct MmRuntimeConfig {
    mode: TransportMode,
    shm_min_bytes: usize,
    log_timing: bool,
    image_encoder_input_dtype: Option<String>,
    rdma: MmRdmaRuntimeConfig,
}

impl Default for MmRuntimeConfig {
    fn default() -> Self {
        Self {
            mode: TransportMode::default(),
            shm_min_bytes: DEFAULT_MULTIMODAL_SHM_MIN_BYTES,
            log_timing: false,
            image_encoder_input_dtype: None,
            rdma: MmRdmaRuntimeConfig::default(),
        }
    }
}

static RUNTIME_CONFIG: OnceLock<MmRuntimeConfig> = OnceLock::new();
static FALLBACK_RUNTIME_CONFIG: OnceLock<MmRuntimeConfig> = OnceLock::new();

/// Seed the process-wide multimodal policy from the validated router config.
/// Call once at startup before serving; idempotent (first call wins).
pub(crate) fn init_mm_runtime_config(config: &RouterConfig) {
    let resolved = MmRuntimeConfig {
        mode: config.multimodal_tensor_transport.unwrap_or_default(),
        shm_min_bytes: config
            .multimodal_shm_min_bytes
            .unwrap_or(DEFAULT_MULTIMODAL_SHM_MIN_BYTES),
        log_timing: config.multimodal_log_timing,
        image_encoder_input_dtype: config.multimodal_image_encoder_input_dtype.clone(),
        rdma: MmRdmaRuntimeConfig {
            listen_ip: config.multimodal_rdma_listen_ip.clone(),
            listen_port: config.multimodal_rdma_listen_port,
            pool_slots: config.multimodal_rdma_pool_slots,
            slot_bytes: config.multimodal_rdma_slot_bytes,
            worker_landing_wait_secs: config.multimodal_rdma_worker_landing_wait_secs,
            worker_read_timeout_secs: config.multimodal_rdma_worker_read_timeout_secs,
            slot_ttl_secs: config.multimodal_rdma_slot_ttl_secs,
        },
    };
    let _ = RUNTIME_CONFIG.set(resolved);
    log_transport_config_once(mm_runtime_config());
}

fn mm_runtime_config() -> &'static MmRuntimeConfig {
    RUNTIME_CONFIG
        .get()
        .unwrap_or_else(|| FALLBACK_RUNTIME_CONFIG.get_or_init(MmRuntimeConfig::default))
}

/// Resolve whether large multimodal tensors should use the SHM transport for
/// this request: per-worker override → router default. `shm` forces SHM whenever
/// SMG can write `/dev/shm` (the operator asserts co-location); `auto` also
/// requires the receiving worker leg to be verified as sharing SMG's `/dev/shm`;
/// `inline` (the default) keeps the gRPC path.
pub(super) fn resolve_mm_shm_enabled(
    workers: Option<&WorkerSelection>,
    skip_pixel_values: bool,
) -> bool {
    let mode = worker_transport_mode_override(workers).unwrap_or_else(|| mm_runtime_config().mode);
    match mode {
        TransportMode::Shm => mm_shm_dev_writable(),
        TransportMode::Auto => {
            worker_shares_dev_shm(workers, skip_pixel_values) && mm_shm_dev_writable()
        }
        // `rdma` routes large tensors through the NIXL pixel lane, not SHM.
        TransportMode::Inline | TransportMode::Rdma => false,
    }
}

/// Resolve the SHM size threshold (bytes) for this request: per-worker override
/// → router default.
pub(super) fn resolve_mm_shm_min_bytes(workers: Option<&WorkerSelection>) -> usize {
    worker_shm_min_bytes_override(workers).unwrap_or_else(|| mm_runtime_config().shm_min_bytes)
}

// ===================== RDMA pixel lane =====================
//
// The gateway owns all RDMA *policy*: it decides whether the lane is on (a
// first-class `TransportMode::Rdma`), builds an injected `RdmaConfig`, and owns
// the single process-wide exporter. The engine-neutral `smg-mm-rdma` crate owns
// only the NIXL mechanics + wire format; it reads no env and no globals. In the
// default (stub) build the exporter is inert, so every export falls back to inline.

/// Fixed agent name the encode worker passes to `fetch_remote_metadata`.
const RDMA_GATEWAY_AGENT_NAME: &str = "smg-gateway-encode";
/// Upper bound on the pre-registered arena (`pool_slots * slot_bytes`). A plausible
/// configuration error (huge pool-slots x slot-bytes) would
/// otherwise flow into a single `vec![0u8; total]` whose allocation failure aborts
/// the process instead of falling back to inline. 8 GiB is well above the 2 GiB
/// default and any realistic pool.
/// Fixed slack added to the derived worker-max-hold when deriving the slot TTL.
/// A const rather than a separate knob: it only ever widens the lost-notif leak window
/// (a capacity nit, never correctness -- the crate's per-lease gen framing makes a
/// recycled-under-read slot detectable independent of the TTL), and 30s dwarfs any
/// Encode-RPC delivery jitter.
const RDMA_SLOT_TTL_SLACK: Duration = Duration::from_secs(30);

/// Process-wide RDMA pixel exporter, built lazily on first use from explicit
/// config when the RDMA lane is enabled. `None` when the lane is off or NIXL init
/// fails (callers then stay on the inline path).
pub(crate) fn mm_rdma_exporter() -> Option<&'static RdmaExporter> {
    static EXPORTER: OnceLock<Option<RdmaExporter>> = OnceLock::new();
    EXPORTER
        .get_or_init(|| {
            if !rdma_lane_enabled() {
                return None;
            }
            let cfg = build_rdma_config(&mm_runtime_config().rdma);
            if cfg.listen_ip.is_empty() {
                // Without a listener IP the worker can't do the cross-node metadata
                // exchange, so every export would fall back to inline anyway. Skip
                // building the NIXL agent + (2 GiB default) arena for nothing.
                warn!(
                    "EPD RDMA: lane enabled without --multimodal-rdma-listen-ip; staying on the inline path"
                );
                return None;
            }
            match RdmaExporter::new(cfg) {
                Ok(exporter) => Some(exporter),
                Err(e) => {
                    error!(error = %e, "EPD RDMA: exporter init failed; inline fallback");
                    None
                }
            }
        })
        .as_ref()
}

/// Whether the RDMA pixel lane is active via the first-class
/// `--multimodal-tensor-transport rdma` policy.
fn rdma_lane_enabled() -> bool {
    mm_runtime_config().mode == TransportMode::Rdma
}

/// Build the exporter config from validated router configuration.
fn build_rdma_config(config: &MmRdmaRuntimeConfig) -> RdmaConfig {
    let slot_bytes = config.slot_bytes.min(MAX_MULTIMODAL_RDMA_ARENA_BYTES);
    let pool_slots = clamp_pool_slots(config.pool_slots, slot_bytes);
    RdmaConfig {
        // Empty listener IP => the exporter cannot do the cross-node metadata
        // exchange, so the caller stays on the inline path (checked before we build
        // the exporter in `mm_rdma_exporter`).
        listen_ip: config.listen_ip.clone().unwrap_or_default(),
        listen_port: config.listen_port,
        agent_name: RDMA_GATEWAY_AGENT_NAME.to_string(),
        pool_slots,
        slot_bytes,
        slot_ttl: derive_rdma_slot_ttl(config),
    }
}

/// Clamp the slot count so the arena (`pool_slots * slot_bytes`) stays within
/// [`MAX_MULTIMODAL_RDMA_ARENA_BYTES`], bounding the startup allocation. Keeps at least one
/// slot; warns when it has to reduce an oversized request.
fn clamp_pool_slots(pool_slots: usize, slot_bytes: usize) -> usize {
    let max_slots = (MAX_MULTIMODAL_RDMA_ARENA_BYTES / slot_bytes.max(1)).max(1);
    if pool_slots > max_slots {
        warn!(
            requested = pool_slots,
            capped = max_slots,
            slot_bytes,
            max_arena_bytes = MAX_MULTIMODAL_RDMA_ARENA_BYTES,
            "EPD RDMA: requested pixel arena exceeds the cap; reducing slot count"
        );
        return max_slots;
    }
    pool_slots
}

/// The worst-case wall time the encode worker may hold a shipped descriptor before
/// and during its one-sided READ: it waits for a landing slot and then performs
/// the READ. These explicit values must match the TokenSpeed worker flags.
fn worker_max_hold(config: &MmRdmaRuntimeConfig) -> Duration {
    Duration::from_secs(
        config
            .worker_landing_wait_secs
            .saturating_add(config.worker_read_timeout_secs),
    )
}

/// How long a leased slot may live without a free-notif before the reaper
/// force-reclaims it. MUST exceed [`worker_max_hold`] or the TTL races a still-valid
/// READ: the reaper frees the slot, the next image re-leases the SAME address, and
/// the late READ silently returns the WRONG image's pixels. Derived by default
/// (= `worker_max_hold` + [`RDMA_SLOT_TTL_SLACK`]); an explicit full-TTL override
/// must exceed the hold (see [`resolve_slot_ttl`]).
fn derive_rdma_slot_ttl(config: &MmRdmaRuntimeConfig) -> Duration {
    resolve_slot_ttl(config.slot_ttl_secs, worker_max_hold(config))
}

/// Apply the TTL invariant to an optional explicit full-TTL override: honor it
/// only if it strictly exceeds `hold` (otherwise the reaper could reclaim a slot the
/// worker is still READing and cross-wire images). A too-small override is ignored
/// with a warning in favor of the derived `hold + RDMA_SLOT_TTL_SLACK`. Pure (takes
/// `hold` as a parameter) so the invariant is unit-tested without touching the env.
fn resolve_slot_ttl(override_secs: Option<u64>, hold: Duration) -> Duration {
    if let Some(secs) = override_secs {
        let ttl = Duration::from_secs(secs);
        if ttl > hold {
            return ttl;
        }
        warn!(
            ttl_s = secs,
            hold_s = hold.as_secs(),
            "multimodal RDMA slot TTL must exceed the worker's max hold; ignoring override"
        );
    }
    hold + RDMA_SLOT_TTL_SLACK
}

fn worker_transport_mode_override(workers: Option<&WorkerSelection>) -> Option<TransportMode> {
    primary_worker(workers)?
        .metadata()
        .spec
        .multimodal_tensor_transport
}

fn worker_shm_min_bytes_override(workers: Option<&WorkerSelection>) -> Option<usize> {
    primary_worker(workers)?
        .metadata()
        .spec
        .multimodal_shm_min_bytes
}

/// The worker whose per-worker overrides apply. Multimodal tensors are sent to
/// wherever the vision encoder runs: the encode worker in EPD (so its spec wins),
/// otherwise the single/prefill worker that does the encoding itself.
fn primary_worker(workers: Option<&WorkerSelection>) -> Option<&Arc<dyn crate::worker::Worker>> {
    match workers? {
        WorkerSelection::Single { worker } => Some(worker),
        WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            ..
        } => encode_assignments
            .as_ref()
            .and_then(|assignments| assignments.first())
            .map(|assignment| &assignment.worker)
            .or(Some(prefill)),
    }
}

pub(super) fn mm_encoder_input_dtype(
    modality: Modality,
    workers: Option<&WorkerSelection>,
) -> String {
    resolve_mm_encoder_input_dtype(
        match modality {
            Modality::Image | Modality::ImageEmbeds => {
                mm_runtime_config().image_encoder_input_dtype.clone()
            }
            Modality::Video | Modality::Audio => None,
        },
        mm_encoder_input_dtype_from_worker(workers),
    )
}

fn resolve_mm_encoder_input_dtype(
    router_override: Option<String>,
    worker_dtype: Option<String>,
) -> String {
    router_override
        .or(worker_dtype)
        .unwrap_or_else(|| "bfloat16".to_string())
}

fn mm_encoder_input_dtype_from_worker(workers: Option<&WorkerSelection>) -> Option<String> {
    primary_worker(workers)?
        .metadata()
        .spec
        .labels
        .get("multimodal_encoder_dtype")
        .filter(|dtype| !dtype.is_empty())
        .cloned()
}

pub(crate) fn log_mm_timing_enabled() -> bool {
    mm_runtime_config().log_timing
}

fn log_transport_config_once(config: &MmRuntimeConfig) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        info!(
            mode = %config.mode,
            shm_min_bytes = config.shm_min_bytes,
            log_timing = config.log_timing,
            image_encoder_input_dtype = ?config.image_encoder_input_dtype,
            dev_writable = mm_shm_dev_writable(),
            "Multimodal tensor transport configured"
        );
    });
}

/// Whether the worker is *verified* to share SMG's `/dev/shm`, making the SHM
/// transport safe for this payload.
///
/// Rather than inferring locality from the worker URL (TCP loopback proves only
/// network locality, not a shared `/dev/shm`), the worker advertises its
/// `/dev/shm` filesystem identity (`<boot_id>:<st_dev of /dev/shm>`) via
/// `GetServerInfo`, which discovery stores in the worker's `shm_namespace_id`
/// label. Two processes share `/dev/shm` iff these tokens match: `boot_id` pins
/// the host, and `st_dev` is the tmpfs superblock device, identical whenever the
/// same tmpfs backs both `/dev/shm` mounts — including separate containers that
/// share it via `--ipc`/bind-mount (where mount-namespace inodes differ but the
/// underlying superblock is the same). We compare the worker's token to ours:
/// equal ⇒ shared. A missing/empty token or any mismatch is treated as
/// non-sharing, so `auto` safely falls back to inline.
fn worker_shares_dev_shm(workers: Option<&WorkerSelection>, skip_pixel_values: bool) -> bool {
    let Some(local) = local_shm_namespace_id() else {
        return false;
    };
    match workers {
        Some(WorkerSelection::Single { worker }) => worker_matches_shm_namespace(worker, local),
        Some(WorkerSelection::Disaggregated {
            encode_assignments,
            prefill,
            decode,
            ..
        }) => {
            if !skip_pixel_values {
                if let Some(encode_assignments) = encode_assignments {
                    // EPD: encoder_input (pixels) ships gateway -> encode worker, so SHM
                    // is safe only if every encode worker assigned in this request shares
                    // the gateway's /dev/shm. A mixed local/remote fan-out must fall back
                    // to inline/RDMA rather than giving a remote worker an unreadable SHM handle.
                    return encode_assignments
                        .iter()
                        .all(|assignment| worker_matches_shm_namespace(&assignment.worker, local));
                }
            }
            worker_matches_shm_namespace(prefill, local)
                && worker_matches_shm_namespace(decode, local)
        }
        None => false,
    }
}

fn worker_matches_shm_namespace(worker: &Arc<dyn crate::worker::Worker>, local: &str) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get("shm_namespace_id")
        .is_some_and(|id| !id.is_empty() && id == local)
}

/// This process's `/dev/shm` filesystem identity: `<boot_id>:<st_dev of /dev/shm>`.
/// `boot_id` pins the host (it is not namespaced) and `st_dev` is the tmpfs
/// superblock device backing `/dev/shm`; together they identify the tmpfs so two
/// processes sharing it (even across containers via `--ipc`/bind-mount) produce
/// the same token. Computed once; `None` if it can't be determined (then `auto`
/// stays inline).
fn local_shm_namespace_id() -> Option<&'static str> {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    ID.get_or_init(compute_shm_namespace_id).as_deref()
}

#[cfg(unix)]
fn compute_shm_namespace_id() -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let shm_dev = std::fs::metadata("/dev/shm").ok()?.dev();
    Some(format!("{}:{shm_dev}", boot_id.trim()))
}

#[cfg(not(unix))]
fn compute_shm_namespace_id() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_dtype_precedes_worker_with_bfloat16_fallback() {
        assert_eq!(
            resolve_mm_encoder_input_dtype(
                Some("float16".to_string()),
                Some("bfloat16".to_string()),
            ),
            "float16"
        );
        assert_eq!(
            resolve_mm_encoder_input_dtype(None, Some("bfloat16".to_string())),
            "bfloat16"
        );
        assert_eq!(resolve_mm_encoder_input_dtype(None, None), "bfloat16");
    }

    /// The derived slot TTL must strictly exceed the worker's max hold, so the
    /// reaper can never reclaim a slot the worker could still be reading (a late
    /// READ against a recycled slot cross-wires images). The crate's `SlotPool`
    /// tests cover the mechanics; this pins the gateway's TTL-derivation policy.
    #[test]
    fn derived_rdma_slot_ttl_exceeds_worker_max_hold() {
        let config = MmRdmaRuntimeConfig::default();
        assert!(
            derive_rdma_slot_ttl(&config) > worker_max_hold(&config),
            "slot_ttl {:?} must exceed worker_max_hold {:?} or a late READ cross-wires",
            derive_rdma_slot_ttl(&config),
            worker_max_hold(&config)
        );
    }

    /// The explicit slot-TTL override is honored only when it exceeds the worker
    /// hold; a too-small (or absent) value falls back to the derived `hold + slack`,
    /// so an operator can never silently reintroduce the recycled-under-READ bug.
    #[test]
    fn slot_ttl_override_must_exceed_hold() {
        let hold = Duration::from_secs(180);
        // Override above the hold is honored verbatim.
        assert_eq!(resolve_slot_ttl(Some(600), hold), Duration::from_secs(600));
        // Override at or below the hold is rejected -> derived hold + slack.
        assert_eq!(
            resolve_slot_ttl(Some(180), hold),
            hold + RDMA_SLOT_TTL_SLACK
        );
        assert_eq!(resolve_slot_ttl(Some(10), hold), hold + RDMA_SLOT_TTL_SLACK);
        // No override -> derived.
        assert_eq!(resolve_slot_ttl(None, hold), hold + RDMA_SLOT_TTL_SLACK);
    }

    /// A slot count that would blow past the arena cap is reduced to fit (>= 1),
    /// while a normal request passes through untouched.
    #[test]
    fn pool_slots_capped_to_arena_max() {
        let slot_bytes = 1024 * 1024 * 1024; // 1 GiB
        let capped = clamp_pool_slots(1_000_000, slot_bytes);
        assert_eq!(capped, MAX_MULTIMODAL_RDMA_ARENA_BYTES / slot_bytes);
        assert!(capped >= 1, "must keep at least one slot");
        assert_eq!(clamp_pool_slots(64, 32 * 1024 * 1024), 64);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn local_shm_namespace_id_resolves_on_linux() {
        // /proc/.../boot_id and /dev/shm both exist on the Linux CI/runtime
        // image, so the token must resolve to `<boot_id>:<st_dev>`. If it ever
        // returned None, `auto` would silently never enable SHM.
        let id = local_shm_namespace_id().expect("shm namespace id should resolve on Linux");
        assert!(
            id.contains(':'),
            "token must be <boot_id>:<st_dev>, got {id:?}"
        );
        let dev = id.rsplit(':').next().unwrap();
        assert!(
            dev.parse::<u64>().is_ok(),
            "st_dev component must be numeric, got {id:?}"
        );
    }
}
