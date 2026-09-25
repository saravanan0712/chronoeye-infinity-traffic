import React, { useState, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import {
    fetchHealthCheck,
    fetchTrafficState,
    fetchTrafficForecasts,
    fetchActiveIncidents,
    fetchCameras,
    fetchGlobalVehicles,
    fetchJourneys,
    fetchReIDJourney,
    fetchActiveVehicles,
} from './services/api';
import {
    NetworkSnapshot,
    SegmentForecast,
    IncidentEvent,
    CameraMetadata,
    VehicleJourney,
    ActiveVehicleTrack,
    IntelligenceKpis,
    SystemStatus,
    GlobalVehicleSummary,
} from './types/dashboard';

import { Header } from './components/Header';
import { IntelligenceKpiPanel } from './components/IntelligenceKpiPanel';
import { CameraGrid } from './components/CameraGrid';
import { JourneyGraph } from './components/JourneyGraph';
import { VehicleInspector } from './components/VehicleInspector';
import { OcrEvidencePanel } from './components/OcrEvidencePanel';
import { SevenSignalPanel } from './components/SevenSignalPanel';
import { TransitionEvidence } from './components/TransitionEvidence';
import { JourneyTimeline } from './components/JourneyTimeline';
import { Module3Handoff } from './components/Module3Handoff';
import { IncidentPanel } from './components/IncidentPanel';
import { TrafficMap } from './components/TrafficMap';
import { ForecastPanel } from './components/ForecastPanel';

import {
    LayoutDashboard,
    Camera,
    Eye,
    Route,
    Fingerprint,
    Network,
    TrendingUp,
    Cpu,
} from 'lucide-react';

type TabType =
    | 'COMMAND_CENTER'
    | 'LIVE_CAMERAS'
    | 'VEHICLE_INTELLIGENCE'
    | 'JOURNEY_RECONSTRUCTION'
    | 'EVIDENCE_ANALYSIS'
    | 'TEMPORAL_NETWORK'
    | 'MODULE3_FORECAST';

export const App: React.FC = () => {
    const { lastMessage } = useWebSocket();
    const [activeTab, setActiveTab] = useState<TabType>('COMMAND_CENTER');
    const [isSimulating, setIsSimulating] = useState<boolean>(true);

    // Backend state & entities
    const [cameras, setCameras] = useState<CameraMetadata[]>([]);
    const [tracksByCamera, setTracksByCamera] = useState<Record<string, ActiveVehicleTrack[]>>({});
    const [globalVehicles, setGlobalVehicles] = useState<GlobalVehicleSummary[]>([]);
    const [journeys, setJourneys] = useState<VehicleJourney[]>([]);
    const [selectedVehicleId, setSelectedVehicleId] = useState<string>('VEH_101');
    const [selectedJourney, setSelectedJourney] = useState<VehicleJourney | null>(null);
    const [selectedSegmentIdx, setSelectedSegmentIdx] = useState<number>(1);
    const [incidents, setIncidents] = useState<IncidentEvent[]>([]);
    const [forecasts, setForecasts] = useState<Record<string, SegmentForecast>>({});
    const [snapshot, setSnapshot] = useState<NetworkSnapshot>({
        timestamp: Date.now() / 1000,
        total_network_vehicles: 42,
        average_network_speed: 48.5,
        network_congestion_index: 0.22,
        segment_states: {},
    });
    const [isDemoMode, setIsDemoMode] = useState<boolean>(false);
    const [healthStatus, setHealthStatus] = useState<string>('OFFLINE');

    // WS payload updates
    useEffect(() => {
        if (lastMessage?.event_type === 'TRAFFIC_STATE_UPDATE') {
            setSnapshot(lastMessage.payload);
        }
    }, [lastMessage]);

    // Fetch initial & recurring data
    useEffect(() => {
        let isMounted = true;

        async function loadAllData() {
            try {
                // 0. Fetch Health Check
                const health = await fetchHealthCheck();
                const isHealthy = health && health.status === 'HEALTHY' && !health._demo;
                if (isMounted) {
                    setHealthStatus(isHealthy ? 'HEALTHY' : 'OFFLINE');
                    if (!isHealthy) setIsDemoMode(true);
                }

                // 1. Fetch Cameras
                const cams = await fetchCameras();
                if (isMounted && cams) {
                    setCameras(cams);
                    if ((cams as any)._demo) setIsDemoMode(true);
                }

                // 2. Fetch Global Vehicles
                const vRes = await fetchGlobalVehicles();
                if (isMounted && vRes && vRes.vehicles) {
                    setGlobalVehicles(vRes.vehicles);
                    if (vRes._demo) setIsDemoMode(true);
                }

                // 3. Fetch Journeys
                const jRes = await fetchJourneys();
                if (isMounted && jRes && jRes.journeys) {
                    setJourneys(jRes.journeys);
                }

                // 4. Fetch Selected Journey
                const activeJ = await fetchReIDJourney(selectedVehicleId);
                if (isMounted && activeJ) {
                    setSelectedJourney(activeJ);
                }

                // 5. Fetch Camera Tracks
                const trackMap: Record<string, ActiveVehicleTrack[]> = {};
                for (const cam of ['CAM_A_EAST', 'CAM_B_WEST', 'CAM_C_NORTH', 'CAM_D_SOUTH']) {
                    const res = await fetchActiveVehicles(cam);
                    if (res && res.vehicles) trackMap[cam] = res.vehicles;
                }
                if (isMounted) setTracksByCamera(trackMap);

                // 6. Traffic Snapshot & Incidents & Forecasts
                const snap = await fetchTrafficState();
                if (isMounted && snap) setSnapshot(snap);

                const incs = await fetchActiveIncidents();
                if (isMounted && incs) setIncidents(incs);

                const fcs = await fetchTrafficForecasts();
                if (isMounted && fcs) setForecasts(fcs);
            } catch (err) {
                console.error('Data load error:', err);
                if (isMounted) {
                    setIsDemoMode(true);
                    setHealthStatus('OFFLINE');
                }
            }
        }

        loadAllData();
        return () => { isMounted = false; };
    }, [selectedVehicleId]);

    // Compute Intelligence KPIs
    const computeKpis = (): IntelligenceKpis => {
        const segs = selectedJourney?.segments || [];
        const confirmedPlates = segs.filter(s => s.plate_status === 'CONFIRMED').length || 8;
        const pendingOcr = segs.filter(s => s.plate_status === 'PENDING').length || 2;
        const unknownPlates = segs.filter(s => s.plate_status === 'UNKNOWN').length || 1;
        const probableTransitions = segs.filter(s => s.transition_decision === 'MATCH_PROBABLE').length || 2;
        const unobservedGaps = segs.filter(s => s.has_unobserved_gap).length || 1;

        const totalTracks = Object.values(tracksByCamera).reduce((acc, curr) => acc + curr.length, 0) || 42;

        return {
            activeCameras: cameras.filter(c => c.status === 'LIVE' || c.status === 'ONLINE').length || 4,
            observedVehicles: totalTracks,
            globalVehicles: globalVehicles.length || 14,
            activeJourneys: journeys.length || 18,
            confirmedPlates: confirmedPlates,
            pendingOcr: pendingOcr,
            unknownPlates: unknownPlates,
            probableTransitions: probableTransitions,
            unobservedGaps: unobservedGaps,
        };
    };

    const kpis = computeKpis();

    const isBackendOnline = healthStatus === 'HEALTHY' && !isDemoMode;
    const hasLiveForecasts = isBackendOnline && Object.keys(forecasts).length > 0;

    const systemStatus: SystemStatus = {
        isBackendOnline,
        isDemoMode,
        module1Status: isBackendOnline ? 'ONLINE' : 'OFFLINE',
        module2Status: isBackendOnline ? 'ONLINE' : 'OFFLINE',
        module3Status: hasLiveForecasts ? 'ONLINE' : isBackendOnline ? 'NOT CONNECTED' : 'OFFLINE',
        activeCameraCount: kpis.activeCameras,
    };

    const tabStyle = (isActive: boolean) => ({
        background: isActive
            ? 'linear-gradient(135deg, rgba(0,212,255,0.22), rgba(0,150,200,0.15))'
            : 'rgba(8,14,26,0.85)',
        color: isActive ? '#fff' : 'var(--text-secondary)',
        border: `1px solid ${isActive ? 'rgba(0,212,255,0.5)' : 'var(--border-color)'}`,
        padding: '0.5rem 1.1rem',
        borderRadius: '8px',
        fontWeight: 700 as const,
        fontSize: '0.7rem' as const,
        letterSpacing: '0.06em' as const,
        cursor: 'pointer' as const,
        display: 'flex' as const,
        alignItems: 'center' as const,
        gap: '0.45rem',
        transition: 'all 0.2s ease',
        boxShadow: isActive ? '0 0 16px rgba(0,212,255,0.15)' : 'none',
        whiteSpace: 'nowrap' as const,
    });

    const segments = selectedJourney?.segments || [];
    const fromSeg = segments[selectedSegmentIdx - 1] || segments[0] || null;
    const toSeg = segments[selectedSegmentIdx] || segments[1] || segments[0] || null;

    return (
        <>
            {/* Aurora Ambient Background */}
            <div className="aurora-bg" />

            <div style={{
                maxWidth: '1540px',
                margin: '0 auto',
                padding: '0.75rem 1.25rem',
                position: 'relative',
                zIndex: 1,
            }}>
                {/* Header */}
                <Header
                    systemStatus={systemStatus}
                    isDemoMode={isDemoMode}
                    isSimulating={isSimulating}
                    onToggleSim={() => setIsSimulating(!isSimulating)}
                    onResetSim={() => setIsSimulating(true)}
                />

                {/* 7-Tab Navigation Bar */}
                <div style={{
                    display: 'flex',
                    gap: '0.45rem',
                    marginBottom: '0.85rem',
                    alignItems: 'center',
                    overflowX: 'auto',
                    paddingBottom: '4px',
                }}>
                    <button
                        id="tab-command-center"
                        onClick={() => setActiveTab('COMMAND_CENTER')}
                        style={tabStyle(activeTab === 'COMMAND_CENTER')}
                    >
                        <LayoutDashboard size={13} />
                        COMMAND CENTER
                    </button>

                    <button
                        id="tab-live-cameras"
                        onClick={() => setActiveTab('LIVE_CAMERAS')}
                        style={tabStyle(activeTab === 'LIVE_CAMERAS')}
                    >
                        <Camera size={13} />
                        LIVE CAMERAS
                    </button>

                    <button
                        id="tab-vehicle-intelligence"
                        onClick={() => setActiveTab('VEHICLE_INTELLIGENCE')}
                        style={tabStyle(activeTab === 'VEHICLE_INTELLIGENCE')}
                    >
                        <Eye size={13} />
                        VEHICLE INTELLIGENCE
                    </button>

                    <button
                        id="tab-journey-reconstruction"
                        onClick={() => setActiveTab('JOURNEY_RECONSTRUCTION')}
                        style={tabStyle(activeTab === 'JOURNEY_RECONSTRUCTION')}
                    >
                        <Route size={13} />
                        JOURNEY RECONSTRUCTION
                    </button>

                    <button
                        id="tab-evidence-analysis"
                        onClick={() => setActiveTab('EVIDENCE_ANALYSIS')}
                        style={tabStyle(activeTab === 'EVIDENCE_ANALYSIS')}
                    >
                        <Fingerprint size={13} />
                        EVIDENCE ANALYSIS
                    </button>

                    <button
                        id="tab-temporal-network"
                        onClick={() => setActiveTab('TEMPORAL_NETWORK')}
                        style={tabStyle(activeTab === 'TEMPORAL_NETWORK')}
                    >
                        <Network size={13} />
                        TEMPORAL NETWORK
                    </button>

                    <button
                        id="tab-module3-forecast"
                        onClick={() => setActiveTab('MODULE3_FORECAST')}
                        style={tabStyle(activeTab === 'MODULE3_FORECAST')}
                    >
                        <TrendingUp size={13} />
                        MODULE 3 FORECAST
                    </button>

                    <div style={{ flex: 1 }} />

                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                        fontSize: '0.62rem',
                        color: 'var(--text-muted)',
                        fontFamily: 'var(--font-mono)',
                        flexShrink: 0,
                    }}>
                        <Cpu size={11} />
                        <span>ChronoEye Infinity v2.0</span>
                    </div>
                </div>

                {/* KPI Panel - always visible */}
                <IntelligenceKpiPanel kpis={kpis} isDemoMode={isDemoMode} />

                {/* TAB 1: COMMAND CENTER */}
                {activeTab === 'COMMAND_CENTER' && (
                    <>
                        {/* Primary: Journey Reconstruction Graph */}
                        <JourneyGraph
                            journey={selectedJourney}
                            selectedSegmentIndex={selectedSegmentIdx}
                            onSelectSegment={(idx) => setSelectedSegmentIdx(idx)}
                            isDemoMode={isDemoMode}
                        />

                        {/* Middle Row: Camera Grid + Vehicle Inspector */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1.2fr 1fr',
                            gap: '0.85rem',
                            marginBottom: '0.85rem',
                        }}>
                            <CameraGrid
                                cameras={cameras}
                                tracksByCamera={tracksByCamera}
                                selectedTrackId={selectedVehicleId}
                                onSelectTrack={(trackId) => setSelectedVehicleId(trackId)}
                                isDemoMode={isDemoMode}
                            />
                            <VehicleInspector
                                initialVehicleId={selectedVehicleId}
                                onSelectJourney={(j) => setSelectedJourney(j)}
                                isDemoMode={isDemoMode}
                            />
                        </div>

                        {/* Bottom Row: Transition Evidence + Incidents */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1.2fr 1fr',
                            gap: '0.85rem',
                        }}>
                            <TransitionEvidence
                                fromSegment={fromSeg}
                                toSegment={toSeg}
                                segmentIndex={selectedSegmentIdx}
                                totalSegments={segments.length}
                                isDemoMode={isDemoMode}
                            />
                            <IncidentPanel incidents={incidents} isDemoMode={isDemoMode} />
                        </div>
                    </>
                )}

                {/* TAB 2: LIVE CAMERAS */}
                {activeTab === 'LIVE_CAMERAS' && (
                    <div>
                        <CameraGrid
                            cameras={cameras}
                            tracksByCamera={tracksByCamera}
                            selectedTrackId={selectedVehicleId}
                            onSelectTrack={(trackId) => setSelectedVehicleId(trackId)}
                            isDemoMode={isDemoMode}
                        />
                        <div style={{ marginTop: '1rem' }}>
                            <VehicleInspector
                                initialVehicleId={selectedVehicleId}
                                onSelectJourney={(j) => setSelectedJourney(j)}
                                isDemoMode={isDemoMode}
                            />
                        </div>
                    </div>
                )}

                {/* TAB 3: VEHICLE INTELLIGENCE */}
                {activeTab === 'VEHICLE_INTELLIGENCE' && (
                    <div>
                        <VehicleInspector
                            initialVehicleId={selectedVehicleId}
                            onSelectJourney={(j) => setSelectedJourney(j)}
                            isDemoMode={isDemoMode}
                        />
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: '0.85rem',
                        }}>
                            <SevenSignalPanel
                                breakdown={toSeg?.transition_breakdown}
                                fromCamera={fromSeg?.camera_id}
                                toCamera={toSeg?.camera_id}
                                isDemoMode={isDemoMode}
                            />
                            <OcrEvidencePanel
                                plateNumber={selectedJourney?.plate_number || undefined}
                                plateStatus={toSeg?.plate_status || undefined}
                                plateConfidence={toSeg?.plate_confidence || undefined}
                                isDemoMode={isDemoMode}
                            />
                        </div>
                    </div>
                )}

                {/* TAB 4: JOURNEY RECONSTRUCTION */}
                {activeTab === 'JOURNEY_RECONSTRUCTION' && (
                    <div>
                        <JourneyGraph
                            journey={selectedJourney}
                            selectedSegmentIndex={selectedSegmentIdx}
                            onSelectSegment={(idx) => setSelectedSegmentIdx(idx)}
                            isDemoMode={isDemoMode}
                        />
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1.2fr',
                            gap: '0.85rem',
                        }}>
                            <JourneyTimeline
                                journey={selectedJourney}
                                selectedSegmentIndex={selectedSegmentIdx}
                                onSelectSegment={(idx) => setSelectedSegmentIdx(idx)}
                                isDemoMode={isDemoMode}
                            />
                            <TransitionEvidence
                                fromSegment={fromSeg}
                                toSegment={toSeg}
                                segmentIndex={selectedSegmentIdx}
                                totalSegments={segments.length}
                                isDemoMode={isDemoMode}
                            />
                        </div>
                    </div>
                )}

                {/* TAB 5: EVIDENCE ANALYSIS */}
                {activeTab === 'EVIDENCE_ANALYSIS' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: '0.85rem',
                        }}>
                            <OcrEvidencePanel
                                plateNumber={selectedJourney?.plate_number || undefined}
                                plateStatus={toSeg?.plate_status || undefined}
                                plateConfidence={toSeg?.plate_confidence || undefined}
                                isDemoMode={isDemoMode}
                            />
                            <SevenSignalPanel
                                breakdown={toSeg?.transition_breakdown}
                                fromCamera={fromSeg?.camera_id}
                                toCamera={toSeg?.camera_id}
                                isDemoMode={isDemoMode}
                            />
                        </div>
                        <TransitionEvidence
                            fromSegment={fromSeg}
                            toSegment={toSeg}
                            segmentIndex={selectedSegmentIdx}
                            totalSegments={segments.length}
                            isDemoMode={isDemoMode}
                        />
                    </div>
                )}

                {/* TAB 6: TEMPORAL NETWORK */}
                {activeTab === 'TEMPORAL_NETWORK' && (
                    <div>
                        <Module3Handoff
                            journeys={journeys}
                            activeVehiclesCount={kpis.observedVehicles}
                            isDemoMode={isDemoMode}
                        />
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '2fr 1fr',
                            gap: '0.85rem',
                            marginTop: '0.85rem',
                        }}>
                            <TrafficMap snapshot={snapshot} incidents={incidents} isDemoMode={isDemoMode} />
                            <IncidentPanel incidents={incidents} isDemoMode={isDemoMode} />
                        </div>
                    </div>
                )}

                {/* TAB 7: MODULE 3 FORECAST */}
                {activeTab === 'MODULE3_FORECAST' && (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: '2fr 1fr',
                        gap: '0.85rem',
                    }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                            <TrafficMap snapshot={snapshot} incidents={incidents} isDemoMode={isDemoMode} />
                            <Module3Handoff
                                journeys={journeys}
                                activeVehiclesCount={kpis.observedVehicles}
                                isDemoMode={isDemoMode}
                            />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                            <ForecastPanel forecasts={forecasts} isDemoMode={isDemoMode} />
                            <IncidentPanel incidents={incidents} isDemoMode={isDemoMode} />
                        </div>
                    </div>
                )}

                {/* Footer */}
                <div style={{
                    marginTop: '1.5rem',
                    padding: '0.65rem 0',
                    borderTop: '1px solid var(--border-color)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '0.6rem',
                    color: 'var(--text-muted)',
                    fontFamily: 'var(--font-mono)',
                }}>
                    <span>CHRONOEYE INFINITY © 2026 — MULTI-SIGNAL VEHICLE-TO-NETWORK INTELLIGENCE</span>
                    <span>SEE → IDENTIFY → TRACK → REMEMBER → UNDERSTAND → PREDICT → DECIDE → OPTIMIZE</span>
                </div>
            </div>
        </>
    );
};

export default App;
