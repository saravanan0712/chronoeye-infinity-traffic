/**
 * ChronoEye Infinity — Dashboard Type Definitions
 * Mirrors backend Pydantic schemas for vehicle intelligence visualization.
 */

// ============================================================
// MATCH DECISION ENUM & PLATE STATUS
// ============================================================

export enum MatchDecision {
    MATCH_CONFIRMED = 'MATCH_CONFIRMED',
    MATCH_PROBABLE = 'MATCH_PROBABLE',
    MATCH_REJECTED = 'MATCH_REJECTED',
    INSUFFICIENT_EVIDENCE = 'INSUFFICIENT_EVIDENCE',
}

export type PlateStatus = 'CONFIRMED' | 'PENDING' | 'UNKNOWN';

// ============================================================
// SEVEN-SIGNAL ReID SCORE BREAKDOWN
// ============================================================

export interface ReIDScoreBreakdown {
    plate_similarity: number;
    appearance_similarity: number;
    visual_features_similarity: number;
    vehicle_type_similarity: number;
    direction_similarity: number;
    temporal_compatibility: number;
    spatial_compatibility: number;
    overall_score: number;
    decision: MatchDecision;
    rejection_reason?: string | null;
}

// ============================================================
// JOURNEY SEGMENT (single camera observation)
// ============================================================

export interface JourneySegment {
    segment_id: string;
    camera_id: string;
    track_id: string;
    timestamp: number;
    speed_estimate: number;
    direction: string;
    plate_number?: string | null;
    plate_confidence?: number | null;
    plate_status?: string | null; // "CONFIRMED" | "PENDING" | "UNKNOWN"
    transition_decision?: MatchDecision | null;
    transition_score?: number | null;
    transition_breakdown?: ReIDScoreBreakdown | null;
    has_unobserved_gap: boolean;
}

// ============================================================
// VEHICLE JOURNEY (reconstructed cross-camera journey)
// ============================================================

export interface VehicleJourney {
    journey_id: string;
    global_vehicle_id: string;
    plate_number?: string | null;
    vehicle_type: string;
    segments: JourneySegment[];
    first_seen: number;
    last_seen: number;
    cameras: string[];
    overall_confidence: number;
    status: string; // "ACTIVE" | "COMPLETED"
}

// ============================================================
// GLOBAL VEHICLE SUMMARY (from /reid/vehicles list)
// ============================================================

export interface GlobalVehicleSummary {
    global_vehicle_id: string;
    journey_id: string;
    plate_number: string;
    vehicle_type: string;
    cameras_visited: string[];
    observation_count: number;
    first_seen: number;
    last_seen: number;
    confidence: number;
    status: string;
}

// ============================================================
// CAMERA NODE (from camera topology)
// ============================================================

export interface CameraNode {
    camera_id: string;
    latitude: number;
    longitude: number;
    road_name: string;
    direction: string;
    connected_cameras: Record<string, number>;
}

// ============================================================
// CAMERA METADATA (from /video/cameras)
// ============================================================

export interface CameraMetadata {
    camera_id: string;
    location_name: string;
    latitude: number;
    longitude: number;
    status: string; // "LIVE" | "PLAYBACK" | "SIMULATION" | "OFFLINE"
    fps: number;
    resolution: string;
    active_source: string;
    road_id: string;
}

// ============================================================
// ACTIVE VEHICLE TRACK (per-camera local track)
// ============================================================

export interface ActiveVehicleTrack {
    track_id: string;
    camera_id: string;
    vehicle_class: string;
    detection_confidence: number;
    tracking_confidence: number;
    speed_kmh: number;
    direction: string;
    first_seen: number;
    last_seen: number;
}

// ============================================================
// FUSED PLATE IDENTITY (OCR evidence)
// ============================================================

export interface FusedPlateIdentity {
    track_id: string;
    camera_id: string;
    best_plate_number: string;
    overall_confidence: number;
    observation_count: number;
    confirmed: boolean;
    status: string; // "CONFIRMED" | "PENDING" | "UNKNOWN"
    character_agreement_ratio: number;
    evidence_frames: number[];
    raw_observations_audit: Array<Record<string, any>>;
}

// ============================================================
// INTELLIGENCE KPIs (computed from backend data)
// ============================================================

export interface IntelligenceKpis {
    activeCameras: number;
    observedVehicles: number;
    globalVehicles: number;
    activeJourneys: number;
    confirmedPlates: number;
    pendingOcr: number;
    unknownPlates: number;
    probableTransitions: number;
    unobservedGaps: number;
}

// ============================================================
// SYSTEM STATUS
// ============================================================

export interface SystemStatus {
    isBackendOnline: boolean;
    isDemoMode: boolean;
    module1Status: 'ONLINE' | 'DEGRADED' | 'OFFLINE' | 'DEMO' | 'NOT_CONNECTED' | 'NOT CONNECTED' | 'DEMO DATA';
    module2Status: 'ONLINE' | 'DEGRADED' | 'OFFLINE' | 'DEMO' | 'NOT_CONNECTED' | 'NOT CONNECTED' | 'DEMO DATA';
    module3Status: 'ONLINE' | 'DEGRADED' | 'OFFLINE' | 'DEMO' | 'NOT_CONNECTED' | 'NOT CONNECTED' | 'DEMO DATA';
    activeCameraCount: number;
}

// ============================================================
// LEGACY TYPES (preserved for existing components)
// ============================================================

export interface RoadSegmentState {
    segment_id: string;
    road_name: string;
    timestamp: number;
    vehicle_count: number;
    flow_rate: number;
    density: number;
    average_speed: number;
    occupancy: number;
    queue_length: number;
    travel_time: number;
    congestion_score: number;
}

export interface NetworkSnapshot {
    timestamp: number;
    segment_states: Record<string, RoadSegmentState>;
    total_network_vehicles: number;
    average_network_speed: number;
    network_congestion_index: number;
}

export interface SegmentForecast {
    segment_id: string;
    horizon: string;
    target_timestamp: number;
    predicted_flow: number;
    predicted_speed: number;
    predicted_density: number;
    predicted_queue_length: number;
    predicted_travel_time: number;
}

export interface IncidentEvent {
    incident_id: string;
    segment_id: string;
    incident_type: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    confidence: number;
    detected_at: number;
    evidence: Record<string, any>;
    status: string;
}

export interface EmergencyCorridorPlan {
    corridor_id: string;
    vehicle_id: string;
    ordered_junctions: string[];
    total_distance_km: number;
    estimated_total_travel_time_seconds: number;
    predicted_arrival_times: Record<string, number>;
    status: 'PLANNING' | 'ACTIVE_GREEN_WAVE' | 'COMPLETED' | 'CANCELLED';
}
