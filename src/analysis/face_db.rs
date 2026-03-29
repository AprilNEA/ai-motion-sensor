use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::geometry::cosine_similarity;

/// Simple in-memory face embedding database.
///
/// Each identity maps to one or more 512-d embeddings (multiple photos of the
/// same person improve robustness).
pub struct FaceDatabase {
    entries: HashMap<String, Vec<Vec<f32>>>,
    threshold: f32,
}

/// Match result.
#[derive(Debug, Clone)]
pub struct FaceMatch {
    pub identity: String,
    pub similarity: f32,
}

impl FaceDatabase {
    pub fn new(threshold: f32) -> Self {
        Self {
            entries: HashMap::new(),
            threshold,
        }
    }

    /// Register an embedding under a given identity.
    pub fn register(&mut self, identity: &str, embedding: Vec<f32>) {
        self.entries
            .entry(identity.to_string())
            .or_default()
            .push(embedding);
    }

    /// Find the best matching identity for a query embedding.
    /// Returns `None` if no match exceeds the similarity threshold.
    pub fn search(&self, query: &[f32]) -> Option<FaceMatch> {
        let mut best: Option<FaceMatch> = None;

        for (name, embeddings) in &self.entries {
            for emb in embeddings {
                let sim = cosine_similarity(query, emb);
                if sim > self.threshold {
                    if best.as_ref().map_or(true, |b| sim > b.similarity) {
                        best = Some(FaceMatch {
                            identity: name.clone(),
                            similarity: sim,
                        });
                    }
                }
            }
        }

        best
    }

    /// Number of registered identities.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Save the database to a simple binary file (identity + embeddings).
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string(&self.entries)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load the database from a previously saved file.
    pub fn load(path: &Path, threshold: f32) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let entries: HashMap<String, Vec<Vec<f32>>> = serde_json::from_str(&data)?;
        Ok(Self { entries, threshold })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_search() {
        let mut db = FaceDatabase::new(0.3);
        let emb_a = vec![1.0; 512];
        let emb_b = vec![-1.0; 512];
        db.register("alice", emb_a.clone());
        db.register("bob", emb_b);

        let result = db.search(&emb_a).unwrap();
        assert_eq!(result.identity, "alice");
        assert!(result.similarity > 0.99);
    }

    #[test]
    fn test_no_match_below_threshold() {
        let mut db = FaceDatabase::new(0.9);
        db.register("alice", vec![1.0; 512]);

        // Orthogonal vector — similarity ≈ 0.
        let mut query = vec![0.0; 512];
        query[0] = 1.0;
        query[1] = -1.0;
        assert!(db.search(&query).is_none());
    }
}
