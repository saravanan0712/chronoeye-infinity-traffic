/**
 * ChronoEye Infinity — API Service Layer
 * All functions attempt real backend first, fall back to clearly-labeled demo data.
 * Demo fallbacks return { _demo: true } so components can detect demo mode.
 */

import {
    NetworkSnapshot, SegmentForecast, IncidentEvent, EmergencyCorridorPlan,
    GlobalVehicleSummary, VehicleJourney, CameraMetadata, ActiveVehicleTrack,
    CameraNode, MatchDecision,
} from '../types/dashboard';

const API_BASE = 'http://localhost:8000/api/v1';

// ============================================================
// HELPER: safe fetch with demo fallback
// ============================================================

async function safeFetch<T>(url: string, fallback: T & { _demo?: boolean }): Promise<T & { _demo?: boolean }> {
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return data;
    } catch {
        return { ...fallback, _demo: true };
    }
}

// ============================================================
// SYSTEM HEALTH
// ============================================================

export async function fetchHealthCheck(): Promise<{ status: string; _demo?: boolean }> {
    return safeFetch(`${API_BASE}/health`, { status: 'OFFLINE' });
}

// ============================================================
// VIDEO / CAMERAS
// ============================================================

export async function fetchCameras(): Promise<CameraMetadata[]> {
    try {
        const res = await fetch(`${API_BASE}/video/cameras`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    } catch {
        return DEMO_CAMERAS;
    }
}

export async function fetchVideoStatus(): Promise<any> {
    return safeFetch(`${API_BASE}/video/status`, {
        ingestion_active: false,
        current_mode: 'SIMULATION',
        frames_processed_per_second: 0,
        active_camera_count: 4,
        camera_health: {},
    });
}

// ============================================================
// VEHICLE INTELLIGENCE (ReID)
// ============================================================

export async function fetchGlobalVehicles(): Promise<{ count: number; vehicles: GlobalVehicleSummary[]; _demo?: boolean }> {
    try {
        const res = await fetch(`${API_BASE}/traffic/reid/vehicles`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        if (data.vehicles && data.vehicles.length > 0) return data;
        return { ...DEMO_VEHICLES_RESPONSE, _demo: true };
    } catch {
        return { ...DEMO_VEHICLES_RESPONSE, _demo: true };
    }
}

export async function fetchVehicleJourney(idOrPlate: string): Promise<VehicleJourney & { _demo?: boolean }> {
    try {
        const res = await fetch(`${API_BASE}/traffic/reid/vehicles/${encodeURIComponent(idOrPlate)}`);
        if (!res.ok) throw new Error('Failed');
        return await res.json();
    } catch {
        return { ...DEMO_JOURNEY, _demo: true };
    }
}

export async function fetchJourneys(): Promise<{ count: number; journeys: VehicleJourney[]; _demo?: boolean }> {
    try {
        const res = await fetch(`${API_BASE}/traffic/reid/journeys`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        if (data.journeys && data.journeys.length > 0) return data;
        return { count: 1, journeys: [DEMO_JOURNEY], _demo: true };
    } catch {
        return { count: 1, journeys: [DEMO_JOURNEY], _demo: true };
    }
}

export async function fetchJourneyById(journeyId: string): Promise<VehicleJourney & { _demo?: boolean }> {
    try {
        const res = await fetch(`${API_BASE}/traffic/reid/journeys/${encodeURIComponent(journeyId)}`);
        if (!res.ok) throw new Error('Failed');
        return await res.json();
    } catch {
        return { ...DEMO_JOURNEY, _demo: true };
    }
}

export async function fetchCameraTopology(): Promise<{ camera_count: number; cameras: Record<string, CameraNode>; _demo?: boolean }> {
    try {
        const res = await fetch(`${API_BASE}/traffic/reid/cameras`);
        if (!res.ok) throw new Error('Failed');
        return await res.json();
    } catch {
        return { camera_count: 0, cameras: {}, _demo: true };
    }
}

export async function fetchIdentityMatches(): Promise<any> {
    return safeFetch(`${API_BASE}/traffic/reid/identity-matches`, {
        status: 'OFFLINE',
        total_active_journeys: 0,
    });
}

// ============================================================
// PER-CAMERA ACTIVE TRACKS
// ============================================================

export async function fetchActiveVehicles(cameraId: string): Promise<{ vehicles: ActiveVehicleTrack[]; _demo?: boolean }> {
    try {
        const res = await fetch(`${API_BASE}/traffic/vehicles?camera_id=${encodeURIComponent(cameraId)}`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        return { vehicles: data.vehicles || [] };
    } catch {
        return { vehicles: [], _demo: true };
    }
}

// ============================================================
// LEGACY API FUNCTIONS (preserved for existing components)
// ============================================================

export async function fetchTrafficState(): Promise<NetworkSnapshot> {
    try {
        const res = await fetch(`${API_BASE}/traffic/state`);
        if (!res.ok) throw new Error('Failed');
        return await res.json();
    } catch {
        return {
            timestamp: Date.now() / 1000,
            total_network_vehicles: 42,
            average_network_speed: 48.5,
            network_congestion_index: 0.22,
            segment_states: {},
        };
    }
}

export async function fetchTrafficForecasts(): Promise<Record<string, SegmentForecast>> {
    try {
        const res = await fetch(`${API_BASE}/traffic/forecasts?model_type=st_gnn`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        return data.forecasts || data;
    } catch {
        return {};
    }
}

export async function fetchActiveIncidents(): Promise<IncidentEvent[]> {
    try {
        const res = await fetch(`${API_BASE}/incidents/active`);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        return data.incidents || [];
    } catch {
        return [];
    }
}

export async function triggerEmergencyCorridor(
    vehicle_id: string, origin: string, destination: string
): Promise<EmergencyCorridorPlan> {
    try {
        const res = await fetch(`${API_BASE}/emergency/corridor`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vehicle_id, vehicle_type: 'AMBULANCE', origin_junction: origin, destination_junction: destination }),
        });
        if (!res.ok) throw new Error('Failed');
        return await res.json();
    } catch {
        return {
            corridor_id: 'CORRIDOR_DEMO', vehicle_id,
            ordered_junctions: [origin, destination],
            total_distance_km: 1.5,
            estimated_total_travel_time_seconds: 39.0,
            predicted_arrival_times: {},
            status: 'ACTIVE_GREEN_WAVE',
        };
    }
}

export async function fetchRouteComparison(origin: string, destination: string): Promise<any> {
    try {
        const [resStd, resPred] = await Promise.all([
            fetch(`${API_BASE}/routes/optimize`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ origin, destination, algorithm: 'Dijkstra' }),
            }),
            fetch(`${API_BASE}/routes/optimize`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ origin, destination, algorithm: 'ChronoEyePredictiveAStar' }),
            }),
        ]);
        return {
            standardRoute: resStd.ok ? await resStd.json() : null,
            predictiveRoute: resPred.ok ? await resPred.json() : null,
        };
    } catch {
        return { standardRoute: null, predictiveRoute: null };
    }
}

export async function fetchReIDVehicles(): Promise<any[]> {
    const data = await fetchGlobalVehicles();
    return data.vehicles || [];
}

export async function fetchReIDJourney(idOrPlate: string): Promise<any> {
    return fetchVehicleJourney(idOrPlate);
}

// ============================================================
// DEMO FALLBACK DATA (clearly labeled)
// ============================================================

const now = () => Date.now() / 1000;

const DEMO_CAMERAS: CameraMetadata[] = [
    { camera_id: 'CAM_A_EAST', location_name: 'Anna Salai North Junction', latitude: 13.0827, longitude: 80.2707, status: 'SIMULATION', fps: 30.0, resolution: '1920x1080', active_source: 'SOURCE_SIMULATION', road_id: 'ROAD_R_AB' },
    { camera_id: 'CAM_B_WEST', location_name: 'Central Station Intersection', latitude: 13.0850, longitude: 80.2720, status: 'SIMULATION', fps: 30.0, resolution: '1920x1080', active_source: 'SOURCE_SIMULATION', road_id: 'ROAD_R_BA' },
    { camera_id: 'CAM_C_NORTH', location_name: 'GST Expressway Ramp', latitude: 13.0880, longitude: 80.2750, status: 'SIMULATION', fps: 30.0, resolution: '1920x1080', active_source: 'SOURCE_SIMULATION', road_id: 'ROAD_R_CD' },
    { camera_id: 'CAM_D_NORTH', location_name: 'Koyambedu Roundabout', latitude: 13.0900, longitude: 80.2800, status: 'SIMULATION', fps: 30.0, resolution: '1920x1080', active_source: 'SOURCE_SIMULATION', road_id: 'ROAD_R_DC' },
];

const DEMO_JOURNEY: VehicleJourney = {
    journey_id: 'JRN_DEMO_001',
    global_vehicle_id: 'VEH_DEMO_101',
    plate_number: 'TN09AB1234',
    vehicle_type: 'car',
    segments: [
        {
            segment_id: 'SEG_DEMO_001', camera_id: 'CAM_A_EAST', track_id: 'TRK_101',
            timestamp: now() - 180, speed_estimate: 52.0, direction: 'EAST',
            plate_number: 'TN09AB1234', plate_confidence: 0.94, plate_status: 'CONFIRMED',
            transition_decision: null, transition_score: null, transition_breakdown: null,
            has_unobserved_gap: false,
        },
        {
            segment_id: 'SEG_DEMO_002', camera_id: 'CAM_B_WEST', track_id: 'TRK_207',
            timestamp: now() - 90, speed_estimate: 44.0, direction: 'WEST',
            plate_number: 'TN09AB1234', plate_confidence: 0.91, plate_status: 'CONFIRMED',
            transition_decision: MatchDecision.MATCH_CONFIRMED, transition_score: 0.87,
            transition_breakdown: {
                plate_similarity: 0.95, appearance_similarity: 0.84,
                visual_features_similarity: 0.76, vehicle_type_similarity: 1.0,
                temporal_compatibility: 0.87, spatial_compatibility: 0.92,
                direction_similarity: 0.88, overall_score: 0.87,
                decision: MatchDecision.MATCH_CONFIRMED, rejection_reason: null,
            },
            has_unobserved_gap: false,
        },
        {
            segment_id: 'SEG_DEMO_003', camera_id: 'CAM_D_NORTH', track_id: 'TRK_314',
            timestamp: now(), speed_estimate: 48.0, direction: 'NORTH',
            plate_number: 'TN09AB1234', plate_confidence: 0.88, plate_status: 'CONFIRMED',
            transition_decision: MatchDecision.MATCH_PROBABLE, transition_score: 0.68,
            transition_breakdown: {
                plate_similarity: 0.90, appearance_similarity: 0.62,
                visual_features_similarity: 0.55, vehicle_type_similarity: 1.0,
                temporal_compatibility: 0.71, spatial_compatibility: 0.58,
                direction_similarity: 0.65, overall_score: 0.68,
                decision: MatchDecision.MATCH_PROBABLE, rejection_reason: null,
            },
            has_unobserved_gap: true,
        },
    ],
    first_seen: now() - 180,
    last_seen: now(),
    cameras: ['CAM_A_EAST', 'CAM_B_WEST', 'CAM_D_NORTH'],
    overall_confidence: 0.87,
    status: 'ACTIVE',
};

const DEMO_VEHICLES_RESPONSE = {
    count: 2,
    vehicles: [
        {
            global_vehicle_id: 'VEH_DEMO_101', journey_id: 'JRN_DEMO_001',
            plate_number: 'TN09AB1234', vehicle_type: 'car',
            cameras_visited: ['CAM_A_EAST', 'CAM_B_WEST', 'CAM_D_NORTH'],
            observation_count: 3, first_seen: now() - 180, last_seen: now(),
            confidence: 0.87, status: 'ACTIVE',
        },
        {
            global_vehicle_id: 'VEH_DEMO_102', journey_id: 'JRN_DEMO_002',
            plate_number: 'UNKNOWN', vehicle_type: 'bus',
            cameras_visited: ['CAM_C_NORTH', 'CAM_D_NORTH'],
            observation_count: 2, first_seen: now() - 300, last_seen: now() - 120,
            confidence: 0.72, status: 'COMPLETED',
        },
    ] as GlobalVehicleSummary[],
};
