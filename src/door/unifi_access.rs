use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::{bail, Context, Result};

use super::{DoorController, DoorLockState};

/// UniFi Access door controller.
///
/// Communicates with the UniFi Access Developer API (port 12445) to
/// list doors and trigger remote unlocks.
pub struct UnifiAccessController {
    host: String,
    token: String,
    client: ureq::Agent,
    /// Cached mapping: door name -> door id.
    door_map: RwLock<HashMap<String, DoorInfo>>,
}

#[derive(Debug, Clone)]
struct DoorInfo {
    id: String,
    is_bind_hub: bool,
    door_lock_relay_status: String,
}

impl UnifiAccessController {
    pub fn new(host: &str, token: &str) -> Result<Self> {
        // UniFi Access uses a self-signed certificate.
        let client = ureq::AgentBuilder::new()
            .tls_config(
                std::sync::Arc::new(
                    rustls::ClientConfig::builder()
                        .dangerous()
                        .with_custom_certificate_verifier(std::sync::Arc::new(NoCertVerifier))
                        .with_no_client_auth(),
                ),
            )
            .build();

        let ctrl = Self {
            host: host.trim_end_matches('/').to_string(),
            token: token.to_string(),
            client,
            door_map: RwLock::new(HashMap::new()),
        };

        ctrl.refresh_doors()?;
        Ok(ctrl)
    }

    /// Fetch all doors from the API and cache them.
    fn refresh_doors(&self) -> Result<()> {
        let url = format!("{}/api/v1/developer/doors", self.host);
        let resp: serde_json::Value = self
            .client
            .get(&url)
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Accept", "application/json")
            .call()
            .context("failed to fetch doors")?
            .into_json()
            .context("failed to parse door list response")?;

        if resp["code"].as_str() != Some("SUCCESS") {
            bail!(
                "UniFi Access API error: {}",
                resp["msg"].as_str().unwrap_or("unknown")
            );
        }

        let data = resp["data"]
            .as_array()
            .context("unexpected door list format")?;

        let mut map = self.door_map.write().unwrap();
        map.clear();
        for door in data {
            let name = door["name"].as_str().unwrap_or_default().to_string();
            let info = DoorInfo {
                id: door["id"].as_str().unwrap_or_default().to_string(),
                is_bind_hub: door["is_bind_hub"].as_bool().unwrap_or(false),
                door_lock_relay_status: door["door_lock_relay_status"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string(),
            };
            tracing::info!(name = %name, id = %info.id, hub = info.is_bind_hub, "discovered door");
            map.insert(name, info);
        }

        Ok(())
    }

    fn find_door(&self, door_name: &str) -> Result<DoorInfo> {
        let map = self.door_map.read().unwrap();
        map.get(door_name)
            .cloned()
            .with_context(|| format!("door '{}' not found in UniFi Access", door_name))
    }
}

impl DoorController for UnifiAccessController {
    fn unlock(&self, door_name: &str) -> Result<()> {
        let door = self.find_door(door_name)?;
        if !door.is_bind_hub {
            bail!("door '{}' is not bound to a hub device", door_name);
        }

        let url = format!(
            "{}/api/v1/developer/doors/{}/unlock",
            self.host, door.id
        );
        let resp: serde_json::Value = self
            .client
            .put(&url)
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/json")
            .set("Accept", "application/json")
            .send_json(serde_json::json!({}))?
            .into_json()
            .context("failed to parse unlock response")?;

        if resp["code"].as_str() != Some("SUCCESS") {
            bail!(
                "unlock failed for '{}': {}",
                door_name,
                resp["msg"].as_str().unwrap_or("unknown")
            );
        }

        tracing::info!(door = door_name, id = %door.id, "door unlocked");
        Ok(())
    }

    fn lock_state(&self, door_name: &str) -> Result<DoorLockState> {
        // Re-fetch to get current state.
        self.refresh_doors()?;
        let door = self.find_door(door_name)?;
        Ok(match door.door_lock_relay_status.as_str() {
            "lock" => DoorLockState::Locked,
            "unlock" => DoorLockState::Unlocked,
            _ => DoorLockState::Unknown,
        })
    }
}

/// Accept any TLS certificate (UniFi Access uses self-signed certs).
#[derive(Debug)]
struct NoCertVerifier;

impl rustls::client::danger::ServerCertVerifier for NoCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls_pki_types::CertificateDer<'_>,
        _intermediates: &[rustls_pki_types::CertificateDer<'_>],
        _server_name: &rustls_pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls_pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls_pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls_pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ED448,
        ]
    }
}
