pub mod unifi_access;

use anyhow::Result;

/// Represents the current state of a door lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoorLockState {
    Locked,
    Unlocked,
    Unknown,
}

/// Trait for controlling a physical door.
///
/// Implementors connect to a door access control system (e.g. UniFi Access)
/// and provide remote lock/unlock capabilities.
pub trait DoorController: Send + Sync {
    /// Unlock the door. Returns immediately after the command is accepted.
    fn unlock(&self, door_name: &str) -> Result<()>;

    /// Query the current lock state, if the backend supports it.
    fn lock_state(&self, door_name: &str) -> Result<DoorLockState>;
}
