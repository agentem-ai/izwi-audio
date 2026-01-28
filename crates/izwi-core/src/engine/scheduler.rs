//! Request scheduler with support for FCFS and priority-based scheduling.
//!
//! The scheduler manages request queues and decides which requests to process
//! in each engine step. It handles:
//! - Waiting queue (new requests awaiting processing)
//! - Running queue (requests currently being processed)
//! - Token budget management
//! - KV cache allocation coordination

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use super::config::EngineCoreConfig;
use super::kv_cache::KVCacheManager;
use super::request::{EngineCoreRequest, RequestStatus};
use super::types::{BlockId, Priority, RequestId, SequenceId};

/// Scheduling policy for the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    /// First-come, first-served (default)
    #[default]
    FCFS,
    /// Priority-based scheduling (higher priority first)
    Priority,
}

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum tokens per step (token budget)
    pub max_tokens_per_step: usize,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,
    /// Threshold for chunked prefill
    pub chunked_prefill_threshold: usize,
    /// Enable preemption when KV cache is full
    pub enable_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_step: 512,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            chunked_prefill_threshold: 256,
            enable_preemption: true,
        }
    }
}

impl From<&EngineCoreConfig> for SchedulerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        Self {
            max_batch_size: config.max_batch_size,
            max_tokens_per_step: config.max_tokens_per_step,
            policy: config.scheduling_policy,
            enable_chunked_prefill: config.enable_chunked_prefill,
            chunked_prefill_threshold: config.chunked_prefill_threshold,
            enable_preemption: config.enable_preemption,
        }
    }
}

/// A request wrapper for priority queue ordering.
#[derive(Debug, Clone)]
struct PriorityRequest {
    request_id: RequestId,
    priority: Priority,
    arrival_time: Instant,
}

impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier arrival time
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.arrival_time.cmp(&self.arrival_time), // Earlier is greater
            ord => ord,
        }
    }
}

/// Result of scheduling a step.
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Requests scheduled for decode (already running)
    pub decode_requests: Vec<ScheduledRequest>,
    /// Requests scheduled for prefill (new requests)
    pub prefill_requests: Vec<ScheduledRequest>,
    /// Requests that were preempted to make room
    pub preempted_requests: Vec<RequestId>,
    /// Total tokens to process this step
    pub total_tokens: usize,
    /// Number of blocks allocated
    pub blocks_allocated: usize,
}

impl ScheduleResult {
    pub fn empty() -> Self {
        Self {
            decode_requests: Vec::new(),
            prefill_requests: Vec::new(),
            preempted_requests: Vec::new(),
            total_tokens: 0,
            blocks_allocated: 0,
        }
    }

    /// Check if there's any work to do
    pub fn has_work(&self) -> bool {
        !self.decode_requests.is_empty() || !self.prefill_requests.is_empty()
    }

    /// Get all scheduled request IDs
    pub fn all_request_ids(&self) -> Vec<RequestId> {
        let mut ids: Vec<_> = self.decode_requests.iter()
            .chain(self.prefill_requests.iter())
            .map(|r| r.request_id.clone())
            .collect();
        ids.dedup();
        ids
    }
}

/// A request that has been scheduled for processing.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Request ID
    pub request_id: RequestId,
    /// Sequence ID
    pub sequence_id: SequenceId,
    /// Number of tokens to process this step
    pub num_tokens: usize,
    /// Whether this is a prefill (first pass) or decode (continuation)
    pub is_prefill: bool,
    /// KV cache blocks allocated to this request
    pub block_ids: Vec<BlockId>,
    /// Number of tokens already computed (for chunked prefill)
    pub num_computed_tokens: usize,
}

/// Request scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    /// Waiting queue (FCFS mode)
    waiting_fcfs: VecDeque<RequestId>,
    /// Waiting queue (Priority mode)
    waiting_priority: BinaryHeap<PriorityRequest>,
    /// Running requests (by request ID)
    running: HashMap<RequestId, RunningRequest>,
    /// Request metadata
    requests: HashMap<RequestId, RequestMetadata>,
    /// Next sequence ID
    next_sequence_id: SequenceId,
}

/// Metadata for a request in the scheduler.
#[derive(Debug, Clone)]
struct RequestMetadata {
    request_id: RequestId,
    sequence_id: SequenceId,
    priority: Priority,
    arrival_time: Instant,
    total_prompt_tokens: usize,
    max_tokens: usize,
}

/// State for a running request.
#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: RequestId,
    sequence_id: SequenceId,
    /// Number of tokens processed so far (prompt + generated)
    num_tokens_processed: usize,
    /// Number of tokens generated so far
    num_tokens_generated: usize,
    /// KV cache blocks allocated
    block_ids: Vec<BlockId>,
    /// Whether prefill is complete
    prefill_complete: bool,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            waiting_fcfs: VecDeque::new(),
            waiting_priority: BinaryHeap::new(),
            running: HashMap::new(),
            requests: HashMap::new(),
            next_sequence_id: 0,
        }
    }

    /// Add a request to the waiting queue.
    pub fn add_request(&mut self, request: &EngineCoreRequest) {
        let sequence_id = self.next_sequence_id;
        self.next_sequence_id += 1;

        let metadata = RequestMetadata {
            request_id: request.id.clone(),
            sequence_id,
            priority: request.priority,
            arrival_time: Instant::now(),
            total_prompt_tokens: request.num_prompt_tokens(),
            max_tokens: request.params.max_tokens,
        };

        self.requests.insert(request.id.clone(), metadata);

        match self.config.policy {
            SchedulingPolicy::FCFS => {
                self.waiting_fcfs.push_back(request.id.clone());
            }
            SchedulingPolicy::Priority => {
                self.waiting_priority.push(PriorityRequest {
                    request_id: request.id.clone(),
                    priority: request.priority,
                    arrival_time: Instant::now(),
                });
            }
        }

        debug!(
            "Added request {} to waiting queue (sequence_id={}, prompt_tokens={})",
            request.id, sequence_id, request.num_prompt_tokens()
        );
    }

    /// Schedule requests for the next step.
    pub fn schedule(&mut self, kv_cache: &mut KVCacheManager) -> ScheduleResult {
        let mut result = ScheduleResult::empty();
        let mut remaining_budget = self.config.max_tokens_per_step;
        let mut remaining_batch = self.config.max_batch_size;

        // Phase 1: Schedule decode requests (already running)
        // Decode requests have priority as they're already using resources
        for (request_id, running) in &self.running {
            if remaining_batch == 0 || remaining_budget == 0 {
                break;
            }

            if !running.prefill_complete {
                continue; // Still in prefill, handle separately
            }

            // Each decode step generates 1 token (or more with speculative decoding)
            let num_tokens = 1;

            // Check if we can allocate more KV cache blocks if needed
            let blocks_needed = self.blocks_needed_for_tokens(running.num_tokens_processed + num_tokens);
            if blocks_needed > running.block_ids.len() {
                let additional = blocks_needed - running.block_ids.len();
                if !kv_cache.can_allocate(additional) {
                    // Try preemption if enabled
                    if self.config.enable_preemption {
                        // TODO: Implement preemption logic
                        warn!("KV cache full, preemption not yet implemented");
                    }
                    continue;
                }
            }

            result.decode_requests.push(ScheduledRequest {
                request_id: request_id.clone(),
                sequence_id: running.sequence_id,
                num_tokens,
                is_prefill: false,
                block_ids: running.block_ids.clone(),
                num_computed_tokens: running.num_tokens_processed,
            });

            remaining_budget = remaining_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            result.total_tokens += num_tokens;
        }

        // Phase 2: Schedule prefill requests (from waiting queue)
        while remaining_batch > 0 && remaining_budget > 0 {
            let next_request_id = match self.config.policy {
                SchedulingPolicy::FCFS => self.waiting_fcfs.front().cloned(),
                SchedulingPolicy::Priority => {
                    self.waiting_priority.peek().map(|r| r.request_id.clone())
                }
            };

            let request_id = match next_request_id {
                Some(id) => id,
                None => break,
            };

            let metadata = match self.requests.get(&request_id) {
                Some(m) => m.clone(),
                None => {
                    self.pop_from_waiting();
                    continue;
                }
            };

            // Check if already running (shouldn't happen, but safety check)
            if self.running.contains_key(&request_id) {
                self.pop_from_waiting();
                continue;
            }

            // Calculate tokens for this prefill
            let mut num_tokens = metadata.total_prompt_tokens;

            // Apply chunked prefill if enabled and prompt is long
            if self.config.enable_chunked_prefill 
                && num_tokens > self.config.chunked_prefill_threshold 
            {
                num_tokens = self.config.chunked_prefill_threshold;
            }

            // Limit by remaining budget
            num_tokens = num_tokens.min(remaining_budget);

            // Allocate KV cache blocks
            let blocks_needed = self.blocks_needed_for_tokens(num_tokens);
            if !kv_cache.can_allocate(blocks_needed) {
                // Can't fit this request, try preemption or skip
                if self.config.enable_preemption {
                    // TODO: Implement preemption
                    warn!("KV cache full for prefill, skipping request {}", request_id);
                }
                break;
            }

            let block_ids = kv_cache.allocate(&request_id, blocks_needed);
            result.blocks_allocated += block_ids.len();

            // Create running state
            let running = RunningRequest {
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens_processed: 0,
                num_tokens_generated: 0,
                block_ids: block_ids.clone(),
                prefill_complete: num_tokens >= metadata.total_prompt_tokens,
            };

            result.prefill_requests.push(ScheduledRequest {
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens,
                is_prefill: true,
                block_ids,
                num_computed_tokens: 0,
            });

            self.running.insert(request_id, running);
            self.pop_from_waiting();

            remaining_budget = remaining_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            result.total_tokens += num_tokens;
        }

        result
    }

    /// Update request state after a step.
    pub fn update_after_step(
        &mut self,
        request_id: &RequestId,
        tokens_processed: usize,
        tokens_generated: usize,
        new_block_ids: Vec<BlockId>,
    ) {
        if let Some(running) = self.running.get_mut(request_id) {
            running.num_tokens_processed += tokens_processed;
            running.num_tokens_generated += tokens_generated;
            running.block_ids.extend(new_block_ids);
            
            // Check if prefill is now complete
            if let Some(metadata) = self.requests.get(request_id) {
                if running.num_tokens_processed >= metadata.total_prompt_tokens {
                    running.prefill_complete = true;
                }
            }
        }
    }

    /// Mark a request as finished and remove it.
    pub fn finish_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) {
        if let Some(running) = self.running.remove(request_id) {
            // Free KV cache blocks
            kv_cache.free(&running.request_id);
            debug!("Finished request {}, freed {} blocks", request_id, running.block_ids.len());
        }
        self.requests.remove(request_id);
    }

    /// Abort a request.
    pub fn abort_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) -> bool {
        // Remove from waiting queue
        self.waiting_fcfs.retain(|id| id != request_id);
        self.waiting_priority.retain(|r| &r.request_id != request_id);

        // Remove from running
        if let Some(running) = self.running.remove(request_id) {
            kv_cache.free(&running.request_id);
            self.requests.remove(request_id);
            return true;
        }

        self.requests.remove(request_id);
        false
    }

    /// Check if a request exists in the scheduler.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Get request status.
    pub fn get_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        if self.running.contains_key(request_id) {
            Some(RequestStatus::Running)
        } else if self.requests.contains_key(request_id) {
            Some(RequestStatus::Waiting)
        } else {
            None
        }
    }

    /// Get number of waiting requests.
    pub fn waiting_count(&self) -> usize {
        match self.config.policy {
            SchedulingPolicy::FCFS => self.waiting_fcfs.len(),
            SchedulingPolicy::Priority => self.waiting_priority.len(),
        }
    }

    /// Get number of running requests.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        self.waiting_count() > 0 || self.running_count() > 0
    }

    /// Get running request info.
    pub fn get_running_info(&self, request_id: &RequestId) -> Option<(usize, usize)> {
        self.running.get(request_id).map(|r| (r.num_tokens_processed, r.num_tokens_generated))
    }

    // Helper methods

    fn pop_from_waiting(&mut self) {
        match self.config.policy {
            SchedulingPolicy::FCFS => { self.waiting_fcfs.pop_front(); }
            SchedulingPolicy::Priority => { self.waiting_priority.pop(); }
        }
    }

    fn blocks_needed_for_tokens(&self, num_tokens: usize) -> usize {
        // Using default block size of 16
        let block_size = 16;
        (num_tokens + block_size - 1) / block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = Scheduler::new(config);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
    }
}
