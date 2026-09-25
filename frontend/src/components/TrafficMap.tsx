import React, { useEffect, useRef, useState, useMemo } from 'react';
import { NetworkSnapshot, IncidentEvent, EmergencyCorridorPlan } from '../types/dashboard';
import {
    Layers,
    Box,
    Flame,
    Zap,
    BarChart3,
    RotateCcw,
    Play,
    Pause,
    FastForward,
    Radio,
    Compass,
    Activity,
    Clock,
    AlertTriangle,
    Eye,
    ChevronRight,
    ArrowUpRight,
} from 'lucide-react';

interface TrafficMapProps {
    snapshot: NetworkSnapshot;
    incidents: IncidentEvent[];
    activeCorridor?: EmergencyCorridorPlan | null;
    isDemoMode?: boolean;
}

type ViewMode = 'NETWORK' | '3D' | 'HEATMAP' | 'FLOW' | 'DENSITY';

// ============================================================
// TOPOLOGY DEFINITION (Preserving existing J1-J5 junctions)
// ============================================================

interface JunctionDef {
    id: string;
    label: string;
    shortLabel: string;
    x: number;
    y: number;
    code: string;
    type: 'ARTERY' | 'HUB' | 'INTERSECTION';
}

const JUNCTIONS: JunctionDef[] = [
    { id: 'JUNC_1', label: 'Anna Salai North',  shortLabel: 'J1', code: 'AS-01', x: 140, y: 110, type: 'ARTERY' },
    { id: 'JUNC_2', label: 'Central Junction',   shortLabel: 'J2', code: 'CJ-02', x: 480, y: 110, type: 'ARTERY' },
    { id: 'JUNC_3', label: 'Bypass Avenue',      shortLabel: 'J3', code: 'BA-03', x: 140, y: 340, type: 'INTERSECTION' },
    { id: 'JUNC_4', label: 'GST Terminal',        shortLabel: 'J4', code: 'GT-04', x: 480, y: 340, type: 'ARTERY' },
    { id: 'JUNC_5', label: 'Tech Park Hub',       shortLabel: 'J5', code: 'TP-05', x: 310, y: 225, type: 'HUB' },
];

interface RoadDef {
    id: string;
    name: string;
    from: string;
    to: string;
    segKey: string;
    lanes: number;
    baseSpeed: number; // km/h
    lengthKm: number;
    hasUnobservedGap?: boolean;
}

const ROADS: RoadDef[] = [
    { id: 'SEG_1_2', name: 'Anna Salai - Central Corridor', from: 'JUNC_1', to: 'JUNC_2', segKey: 'ROAD_R_AB', lanes: 3, baseSpeed: 52, lengthKm: 1.8 },
    { id: 'SEG_2_4', name: 'Central - GST Arterial',         from: 'JUNC_2', to: 'JUNC_4', segKey: 'ROAD_R_BC', lanes: 2, baseSpeed: 46, lengthKm: 2.1 },
    { id: 'SEG_1_3', name: 'West Outer Connector',           from: 'JUNC_1', to: 'JUNC_3', segKey: 'ROAD_R_CD', lanes: 2, baseSpeed: 50, lengthKm: 1.9 },
    { id: 'SEG_3_4', name: 'Bypass South Corridor',          from: 'JUNC_3', to: 'JUNC_4', segKey: 'ROAD_R_DE', lanes: 3, baseSpeed: 55, lengthKm: 2.4 },
    { id: 'SEG_1_5', name: 'North Hub Feeder',               from: 'JUNC_1', to: 'JUNC_5', segKey: 'ROAD_R_AE', lanes: 2, baseSpeed: 44, lengthKm: 1.4 },
    { id: 'SEG_5_4', name: 'Hub - GST Expressway',          from: 'JUNC_5', to: 'JUNC_4', segKey: 'ROAD_R_EF', lanes: 2, baseSpeed: 48, lengthKm: 1.5 },
    { id: 'SEG_5_2', name: 'Hub - Central Link',             from: 'JUNC_5', to: 'JUNC_2', segKey: 'ROAD_R_EG', lanes: 1, baseSpeed: 40, lengthKm: 1.2 },
    { id: 'SEG_5_3', name: 'Hub - Bypass Link',              from: 'JUNC_5', to: 'JUNC_3', segKey: 'ROAD_R_EH', lanes: 1, baseSpeed: 42, lengthKm: 1.3 },
    { id: 'SEG_GAP_1_4', name: 'Blind Zone Sector 4 (Unobserved)', from: 'JUNC_1', to: 'JUNC_4', segKey: 'ROAD_R_UNOBSERVED', lanes: 1, baseSpeed: 35, lengthKm: 3.2, hasUnobservedGap: true },
];

const SIGNAL_DEFAULTS: Record<string, 'GREEN' | 'RED' | 'AMBER'> = {
    JUNC_1: 'GREEN',
    JUNC_2: 'RED',
    JUNC_3: 'GREEN',
    JUNC_4: 'GREEN',
    JUNC_5: 'AMBER',
};

// Subtle schematic city buildings footprint polygons for digital twin background
const CITY_BUILDINGS = [
    { points: '60,50 110,40 115,80 65,90', id: 'b1' },
    { points: '165,45 220,40 225,85 170,90', id: 'b2' },
    { points: '390,40 450,45 445,85 385,80', id: 'b3' },
    { points: '500,50 560,45 555,90 495,95', id: 'b4' },
    { points: '50,150 100,160 95,210 45,200', id: 'b5' },
    { points: '520,160 570,150 575,210 525,220', id: 'b6' },
    { points: '55,270 105,260 110,310 60,320', id: 'b7' },
    { points: '510,270 565,265 570,315 515,320', id: 'b8' },
    { points: '170,370 230,365 225,405 165,410', id: 'b9' },
    { points: '380,370 440,365 445,410 385,415', id: 'b10' },
    { points: '240,135 285,130 280,165 235,170', id: 'b11' },
    { points: '335,130 380,135 375,170 330,165', id: 'b12' },
    { points: '240,285 285,280 280,320 235,325', id: 'b13' },
    { points: '335,280 380,285 375,325 330,320', id: 'b14' },
];

const congestionColor = (score: number) => {
    if (score >= 0.7) return '#ff1860'; // Congested (Magenta/Red)
    if (score >= 0.4) return '#ffb800'; // Moderate (Amber)
    return '#00ff9d'; // Free Flow (Neon Green/Cyan)
};

const congestionLabel = (score: number) => {
    if (score >= 0.7) return 'CONGESTED';
    if (score >= 0.4) return 'MODERATE';
    return 'FREE FLOW';
};

interface FlowParticle {
    id: string;
    roadId: string;
    progress: number;
    speed: number;
    color: string;
    size: number;
}

export const TrafficMap: React.FC<TrafficMapProps> = ({
    snapshot,
    incidents,
    activeCorridor,
    isDemoMode = false,
}) => {
    const [viewMode, setViewMode] = useState<ViewMode>('NETWORK');
    const [selectedJunctionId, setSelectedJunctionId] = useState<string | null>(null);
    const [hoveredJunctionId, setHoveredJunctionId] = useState<string | null>(null);
    const [selectedRoadId, setSelectedRoadId] = useState<string | null>(null);
    const [hoveredRoadId, setHoveredRoadId] = useState<string | null>(null);
    const [showUnobserved, setShowUnobserved] = useState<boolean>(true);

    // Simulation & Playback State
    const [isPlaying, setIsPlaying] = useState<boolean>(true);
    const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
    const [playbackTimeIdx, setPlaybackTimeIdx] = useState<number>(3); // 10:15
    const [signals, setSignals] = useState(SIGNAL_DEFAULTS);
    const [particles, setParticles] = useState<FlowParticle[]>([]);

    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const junctionMap = useMemo(() => Object.fromEntries(JUNCTIONS.map(j => [j.id, j])), []);

    // Visible roads filter
    const visibleRoads = useMemo(() => {
        return ROADS.filter(r => !r.hasUnobservedGap || showUnobserved);
    }, [showUnobserved]);

    // Initialize Flow Particles
    useEffect(() => {
        const pts: FlowParticle[] = [];
        ROADS.forEach((r, ri) => {
            const count = r.hasUnobservedGap ? 2 : Math.floor(r.lanes * 2.5);
            for (let i = 0; i < count; i++) {
                pts.push({
                    id: `P_${ri}_${i}`,
                    roadId: r.id,
                    progress: Math.random(),
                    speed: 0.003 + Math.random() * 0.005,
                    color: r.hasUnobservedGap ? '#a855f7' : '#00d4ff',
                    size: r.hasUnobservedGap ? 2.5 : 3.5,
                });
            }
        });
        setParticles(pts);
    }, []);

    // Animation Loop
    useEffect(() => {
        if (!isPlaying) return;
        let animId: number;

        const animate = () => {
            setParticles(prev =>
                prev.map(p => {
                    const road = ROADS.find(r => r.id === p.roadId);
                    const congScore = road ? (snapshot.segment_states[road.segKey]?.congestion_score ?? 0.3) : 0.3;
                    // Congested vehicles move slower
                    const speedModifier = congScore > 0.7 ? 0.45 : congScore > 0.4 ? 0.8 : 1.3;
                    const delta = p.speed * speedModifier * (viewMode === 'FLOW' ? 1.8 : 1.0) * playbackSpeed;
                    return {
                        ...p,
                        progress: (p.progress + delta) % 1,
                    };
                })
            );
            animId = requestAnimationFrame(animate);
        };

        animId = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animId);
    }, [isPlaying, playbackSpeed, snapshot, viewMode]);

    // Cycle signal phases every 6 seconds
    useEffect(() => {
        if (!isPlaying) return;
        const timer = setInterval(() => {
            setSignals(prev => {
                const next = { ...prev };
                Object.keys(next).forEach(jId => {
                    if (next[jId] === 'GREEN') next[jId] = 'AMBER';
                    else if (next[jId] === 'AMBER') next[jId] = 'RED';
                    else if (next[jId] === 'RED') next[jId] = 'GREEN';
                });
                return next;
            });
        }, 6000 / playbackSpeed);
        return () => clearInterval(timer);
    }, [isPlaying, playbackSpeed]);

    // Active Corridor Detection
    const isEmergencyRoad = (roadId: string) => {
        if (!activeCorridor?.ordered_junctions) return false;
        const road = ROADS.find(r => r.id === roadId);
        if (!road) return false;
        const junctions = activeCorridor.ordered_junctions;
        const fromIdx = junctions.indexOf(road.from);
        const toIdx = junctions.indexOf(road.to);
        return fromIdx !== -1 && toIdx !== -1 && Math.abs(fromIdx - toIdx) === 1;
    };

    const isEmergencyJunction = (jId: string) =>
        activeCorridor?.ordered_junctions?.includes(jId) ?? false;

    // Node & Edge selection helpers
    const activeJunction = selectedJunctionId || hoveredJunctionId;
    const activeRoad = selectedRoadId || hoveredRoadId;

    const isNodeConnectedToActive = (jId: string) => {
        if (!activeJunction) return false;
        if (jId === activeJunction) return true;
        return ROADS.some(r => (r.from === activeJunction && r.to === jId) || (r.to === activeJunction && r.from === jId));
    };

    const isRoadConnectedToActive = (r: RoadDef) => {
        if (activeRoad) return r.id === activeRoad;
        if (!activeJunction) return true;
        return r.from === activeJunction || r.to === activeJunction;
    };

    // Calculate Particle Position
    const getParticlePos = (p: FlowParticle) => {
        const road = ROADS.find(r => r.id === p.roadId);
        if (!road) return null;
        const u = junctionMap[road.from];
        const v = junctionMap[road.to];
        if (!u || !v) return null;

        // Subtle curved path offset for unobserved sector
        if (road.hasUnobservedGap) {
            const midX = (u.x + v.x) / 2 - 40;
            const midY = (u.y + v.y) / 2 + 50;
            const t = p.progress;
            const x = (1 - t) * (1 - t) * u.x + 2 * (1 - t) * t * midX + t * t * v.x;
            const y = (1 - t) * (1 - t) * u.y + 2 * (1 - t) * t * midY + t * t * v.y;
            return { x, y, road };
        }

        return {
            x: u.x + (v.x - u.x) * p.progress,
            y: u.y + (v.y - u.y) * p.progress,
            road,
        };
    };

    // Derived dynamic junction stats
    const getJunctionMetrics = (juncId: string) => {
        const connected = ROADS.filter(r => r.from === juncId || r.to === juncId);
        const inbound = ROADS.filter(r => r.to === juncId).length;
        const outbound = ROADS.filter(r => r.from === juncId).length;
        const avgCongestion = connected.reduce((acc, r) => {
            const c = snapshot.segment_states[r.segKey]?.congestion_score ?? 0.35;
            return acc + c;
        }, 0) / (connected.length || 1);

        const flowRate = Math.round(320 + avgCongestion * 280);
        const avgSpeed = Math.round(54 - avgCongestion * 26);
        const density = (0.35 + avgCongestion * 0.55).toFixed(2);
        const travelTime = (2.4 + avgCongestion * 3.8).toFixed(1);

        return {
            flowRate,
            avgSpeed,
            density,
            travelTime,
            avgCongestion,
            status: congestionLabel(avgCongestion),
            color: congestionColor(avgCongestion),
            inbound,
            outbound,
        };
    };

    // Active hovered/selected info
    const inspectedJunction = activeJunction ? {
        ...junctionMap[activeJunction],
        metrics: getJunctionMetrics(activeJunction),
    } : null;

    const inspectedRoad = activeRoad ? ROADS.find(r => r.id === activeRoad) : null;

    // Timeline steps
    const TIMELINE_STEPS = ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'];

    return (
        <div
            ref={containerRef}
            className="glass-panel"
            style={{
                padding: '1.25rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '1rem',
                position: 'relative',
                background: 'rgba(6, 10, 20, 0.94)',
                border: '1px solid rgba(0, 212, 255, 0.2)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
            }}
        >
            {/* ============================================================
                1. HEADER & LEGEND BAR
                ============================================================ */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
                <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <div style={{
                            width: '32px', height: '32px', borderRadius: '8px',
                            background: 'radial-gradient(circle, rgba(0,212,255,0.2) 0%, rgba(4,7,14,0.8) 100%)',
                            border: '1px solid rgba(0,212,255,0.4)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            boxShadow: '0 0 14px rgba(0,212,255,0.3)',
                        }}>
                            <Layers size={18} color="var(--accent-cyan)" />
                        </div>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <h2 style={{
                                    fontFamily: 'var(--font-display)',
                                    fontSize: '0.85rem',
                                    fontWeight: 800,
                                    color: 'var(--text-primary)',
                                    letterSpacing: '0.12em',
                                    textShadow: '0 0 10px rgba(0,212,255,0.4)',
                                }}>
                                    SPATIO-TEMPORAL TRAFFIC GRAPH
                                </h2>
                                <span className={`demo-mode-badge ${isDemoMode ? 'demo' : 'live'}`} style={{ fontSize: '0.55rem', padding: '2px 7px' }}>
                                    <span style={{
                                        width: '5px', height: '5px', borderRadius: '50%',
                                        background: isDemoMode ? 'var(--accent-amber)' : 'var(--accent-green)',
                                        display: 'inline-block', marginRight: '4px',
                                    }} />
                                    {isDemoMode ? 'DEMO DATA' : 'LIVE DATA'}
                                </span>
                            </div>
                            <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>
                                City Network • Real-time Flow • Travel Time • Congestion State • Spatio-Temporal Twin
                            </p>
                        </div>
                    </div>
                </div>

                {/* Legend */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    background: 'var(--bg-inset)',
                    padding: '0.35rem 0.75rem',
                    borderRadius: '6px',
                    border: '1px solid rgba(0,212,255,0.1)',
                    fontSize: '0.62rem',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-secondary)',
                    flexWrap: 'wrap',
                }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#00ff9d', boxShadow: '0 0 6px #00ff9d' }} />
                        FREE FLOW
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#ffb800', boxShadow: '0 0 6px #ffb800' }} />
                        MODERATE
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#ff1860', boxShadow: '0 0 6px #ff1860' }} />
                        CONGESTED
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#00d4ff', boxShadow: '0 0 6px #00d4ff' }} />
                        CAMERA NODE
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: 7, height: 7, borderRadius: '2px', border: '1px dashed #a855f7', background: 'rgba(168,85,247,0.3)' }} />
                        UNOBSERVED
                    </span>
                </div>
            </div>

            {/* ============================================================
                2. VIEW MODE TOOLBAR & CONTROLS
                ============================================================ */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '0.5rem',
                flexWrap: 'wrap',
                background: 'rgba(4, 7, 14, 0.85)',
                padding: '0.4rem 0.6rem',
                borderRadius: '8px',
                border: '1px solid rgba(0,212,255,0.15)',
            }}>
                <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                    {[
                        { id: 'NETWORK', label: 'NETWORK VIEW', icon: <Layers size={12} /> },
                        { id: '3D', label: '3D PERSPECTIVE', icon: <Box size={12} /> },
                        { id: 'HEATMAP', label: 'HEATMAP', icon: <Flame size={12} /> },
                        { id: 'FLOW', label: 'FLOW ANIMATION', icon: <Zap size={12} /> },
                        { id: 'DENSITY', label: 'TRAFFIC DENSITY', icon: <BarChart3 size={12} /> },
                    ].map(btn => {
                        const isActive = viewMode === btn.id;
                        return (
                            <button
                                key={btn.id}
                                onClick={() => setViewMode(btn.id as ViewMode)}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '0.35rem',
                                    background: isActive
                                        ? 'linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,120,200,0.15))'
                                        : 'rgba(8,14,26,0.6)',
                                    color: isActive ? '#00d4ff' : 'var(--text-secondary)',
                                    border: `1px solid ${isActive ? 'rgba(0,212,255,0.6)' : 'rgba(255,255,255,0.06)'}`,
                                    borderRadius: '5px',
                                    padding: '0.3rem 0.65rem',
                                    fontSize: '0.62rem',
                                    fontWeight: 700,
                                    fontFamily: 'var(--font-mono)',
                                    letterSpacing: '0.04em',
                                    cursor: 'pointer',
                                    transition: 'all 0.15s ease',
                                    boxShadow: isActive ? '0 0 10px rgba(0,212,255,0.2)' : 'none',
                                }}
                            >
                                {btn.icon}
                                {btn.label}
                            </button>
                        );
                    })}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    {/* Unobserved Toggle */}
                    <button
                        onClick={() => setShowUnobserved(!showUnobserved)}
                        style={{
                            background: showUnobserved ? 'rgba(168,85,247,0.18)' : 'rgba(8,14,26,0.6)',
                            color: showUnobserved ? 'var(--accent-purple)' : 'var(--text-muted)',
                            border: `1px solid ${showUnobserved ? 'rgba(168,85,247,0.4)' : 'rgba(255,255,255,0.06)'}`,
                            borderRadius: '5px',
                            padding: '0.3rem 0.6rem',
                            fontSize: '0.6rem',
                            fontFamily: 'var(--font-mono)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                        }}
                    >
                        <Eye size={11} />
                        UNOBSERVED SECTOR
                    </button>

                    {/* Reset View */}
                    <button
                        onClick={() => {
                            setViewMode('NETWORK');
                            setSelectedJunctionId(null);
                            setSelectedRoadId(null);
                        }}
                        style={{
                            background: 'rgba(8,14,26,0.6)',
                            color: 'var(--text-secondary)',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: '5px',
                            padding: '0.3rem 0.6rem',
                            fontSize: '0.6rem',
                            fontFamily: 'var(--font-mono)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                        }}
                        title="Reset selection & view"
                    >
                        <RotateCcw size={11} />
                        RESET
                    </button>
                </div>
            </div>

            {/* ============================================================
                3. MAIN DIGITAL-TWIN GRAPH VIEWPORT
                ============================================================ */}
            <div style={{
                position: 'relative',
                borderRadius: '10px',
                overflow: 'hidden',
                background: '#020812',
                border: '1px solid rgba(0, 212, 255, 0.25)',
                boxShadow: 'inset 0 0 40px rgba(0, 30, 60, 0.6)',
                perspective: viewMode === '3D' ? '1000px' : 'none',
            }}>
                {/* Floating GPS HUD coordinates */}
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '12px',
                    zIndex: 5,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.55rem',
                    color: 'rgba(0, 212, 255, 0.6)',
                    pointerEvents: 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '2px',
                }}>
                    <span>LAT 13.0827° N · LON 80.2707° E</span>
                    <span style={{ color: 'rgba(120, 140, 170, 0.6)' }}>GRID SECTOR: CHENNAI_METRO_CORE</span>
                </div>

                {/* View Mode Watermark Tag */}
                <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '12px',
                    zIndex: 5,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.58rem',
                    fontWeight: 700,
                    color: 'var(--accent-cyan)',
                    background: 'rgba(4, 7, 14, 0.8)',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    border: '1px solid rgba(0,212,255,0.3)',
                    pointerEvents: 'none',
                }}>
                    MODE: {viewMode}
                </div>

                {/* SVG Canvas Container */}
                <div style={{
                    transform: viewMode === '3D' ? 'rotateX(22deg) rotateZ(-2deg) scale(0.96)' : 'none',
                    transformOrigin: '50% 50%',
                    transition: 'transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1)',
                }}>
                    <svg
                        ref={svgRef}
                        width="100%"
                        height="430"
                        viewBox="0 0 620 440"
                        preserveAspectRatio="xMidYMid meet"
                        onClick={() => {
                            setSelectedJunctionId(null);
                            setSelectedRoadId(null);
                        }}
                        style={{ display: 'block' }}
                    >
                        {/* SVG DEFS & SHADERS */}
                        <defs>
                            {/* Grid Pattern */}
                            <pattern id="dt-grid-fine" width="20" height="20" patternUnits="userSpaceOnUse">
                                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0, 212, 255, 0.035)" strokeWidth="0.5" />
                            </pattern>
                            <pattern id="dt-grid-major" width="100" height="100" patternUnits="userSpaceOnUse">
                                <path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(0, 212, 255, 0.08)" strokeWidth="0.8" />
                                <circle cx="0" cy="0" r="1.5" fill="rgba(0,212,255,0.3)" />
                                <circle cx="100" cy="0" r="1.5" fill="rgba(0,212,255,0.3)" />
                                <circle cx="0" cy="100" r="1.5" fill="rgba(0,212,255,0.3)" />
                                <circle cx="100" cy="100" r="1.5" fill="rgba(0,212,255,0.3)" />
                            </pattern>

                            {/* Glow Filters */}
                            <filter id="dt-glow-subtle" x="-20%" y="-20%" width="140%" height="140%">
                                <feGaussianBlur stdDeviation="2.5" result="blur" />
                                <feMerge>
                                    <feMergeNode in="blur" />
                                    <feMergeNode in="SourceGraphic" />
                                </feMerge>
                            </filter>
                            <filter id="dt-glow-strong" x="-40%" y="-40%" width="180%" height="180%">
                                <feGaussianBlur stdDeviation="5.5" result="blur" />
                                <feMerge>
                                    <feMergeNode in="blur" />
                                    <feMergeNode in="SourceGraphic" />
                                </feMerge>
                            </filter>
                            <filter id="dt-heat-blur" x="-50%" y="-50%" width="200%" height="200%">
                                <feGaussianBlur stdDeviation="12" result="heatBlur" />
                                <feMerge>
                                    <feMergeNode in="heatBlur" />
                                    <feMergeNode in="SourceGraphic" />
                                </feMerge>
                            </filter>

                            {/* Gradients */}
                            <radialGradient id="hub-radial-glow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stopColor="rgba(139, 0, 255, 0.35)" />
                                <stop offset="60%" stopColor="rgba(0, 212, 255, 0.15)" />
                                <stop offset="100%" stopColor="rgba(0, 212, 255, 0)" />
                            </radialGradient>
                            <radialGradient id="radar-sweep" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stopColor="rgba(0, 212, 255, 0.08)" />
                                <stop offset="100%" stopColor="rgba(0, 212, 255, 0)" />
                            </radialGradient>
                        </defs>

                        {/* DIGITAL CITY BACKGROUND */}
                        <rect width="620" height="440" fill="#020812" />
                        <rect width="620" height="440" fill="url(#dt-grid-fine)" />
                        <rect width="620" height="440" fill="url(#dt-grid-major)" />

                        {/* Radar concentric rings centered on J5 (Tech Park Hub) */}
                        <g opacity="0.4" pointerEvents="none">
                            <circle cx="310" cy="225" r="90" fill="none" stroke="rgba(0, 212, 255, 0.08)" strokeDasharray="3 4" />
                            <circle cx="310" cy="225" r="170" fill="none" stroke="rgba(0, 212, 255, 0.06)" strokeDasharray="4 6" />
                            <circle cx="310" cy="225" r="250" fill="none" stroke="rgba(0, 212, 255, 0.04)" />
                            <line x1="310" y1="20" x2="310" y2="430" stroke="rgba(0,212,255,0.04)" strokeDasharray="2 4" />
                            <line x1="20" y1="225" x2="600" y2="225" stroke="rgba(0,212,255,0.04)" strokeDasharray="2 4" />
                        </g>

                        {/* City Building Footprints Wireframes */}
                        <g opacity="0.35" pointerEvents="none">
                            {CITY_BUILDINGS.map(b => (
                                <polygon
                                    key={b.id}
                                    points={b.points}
                                    fill="rgba(0, 30, 60, 0.3)"
                                    stroke="rgba(0, 212, 255, 0.12)"
                                    strokeWidth="0.8"
                                />
                            ))}
                        </g>

                        {/* HEATMAP LAYER (when HEATMAP mode is active) */}
                        {viewMode === 'HEATMAP' && (
                            <g filter="url(#dt-heat-blur)" opacity="0.65" pointerEvents="none">
                                {visibleRoads.map(road => {
                                    const u = junctionMap[road.from];
                                    const v = junctionMap[road.to];
                                    if (!u || !v) return null;
                                    const cong = snapshot.segment_states[road.segKey]?.congestion_score ?? 0.45;
                                    const heatColor = congestionColor(cong);
                                    return (
                                        <line
                                            key={`heat_${road.id}`}
                                            x1={u.x} y1={u.y} x2={v.x} y2={v.y}
                                            stroke={heatColor}
                                            strokeWidth={20 + cong * 24}
                                            strokeLinecap="round"
                                        />
                                    );
                                })}
                                {JUNCTIONS.map(j => {
                                    const m = getJunctionMetrics(j.id);
                                    return (
                                        <circle
                                            key={`heat_node_${j.id}`}
                                            cx={j.x} cy={j.y}
                                            r={35 + m.avgCongestion * 30}
                                            fill={m.color}
                                        />
                                    );
                                })}
                            </g>
                        )}

                        {/* ============================================================
                            ROAD / EDGE RENDERING
                            ============================================================ */}
                        {visibleRoads.map(road => {
                            const u = junctionMap[road.from];
                            const v = junctionMap[road.to];
                            if (!u || !v) return null;

                            const congScore = snapshot.segment_states[road.segKey]?.congestion_score ?? (road.hasUnobservedGap ? 0.3 : 0.42);
                            const isEmergency = isEmergencyRoad(road.id);
                            const isSelected = selectedRoadId === road.id || hoveredRoadId === road.id;
                            const isConnected = isRoadConnectedToActive(road);
                            const baseColor = isEmergency ? '#00d4ff' : road.hasUnobservedGap ? '#a855f7' : congestionColor(congScore);

                            // Variable Width calculation
                            const densityMultiplier = viewMode === 'DENSITY' ? 1.5 : 1.0;
                            const strokeW = road.hasUnobservedGap
                                ? 2.5
                                : (2 + road.lanes * 2.2 + congScore * 4) * densityMultiplier;

                            // Opacity for network focus filtering
                            const edgeOpacity = activeJunction || activeRoad ? (isConnected ? 1.0 : 0.28) : 0.85;

                            // Curvature for unobserved bypass
                            if (road.hasUnobservedGap) {
                                const midX = (u.x + v.x) / 2 - 40;
                                const midY = (u.y + v.y) / 2 + 50;
                                const pathD = `M ${u.x} ${u.y} Q ${midX} ${midY} ${v.x} ${v.y}`;

                                return (
                                    <g
                                        key={road.id}
                                        opacity={edgeOpacity}
                                        style={{ cursor: 'pointer', transition: 'opacity 0.2s ease' }}
                                        onMouseEnter={() => setHoveredRoadId(road.id)}
                                        onMouseLeave={() => setHoveredRoadId(null)}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setSelectedRoadId(road.id);
                                            setSelectedJunctionId(null);
                                        }}
                                    >
                                        {/* Underglow */}
                                        <path
                                            d={pathD}
                                            fill="none"
                                            stroke="#a855f7"
                                            strokeWidth="10"
                                            strokeOpacity="0.12"
                                        />
                                        {/* Dashed Unobserved Arc */}
                                        <path
                                            d={pathD}
                                            fill="none"
                                            stroke="#a855f7"
                                            strokeWidth="2.5"
                                            strokeDasharray="6 6"
                                            filter="url(#dt-glow-subtle)"
                                        />
                                        {/* Label Badge */}
                                        <g transform={`translate(${midX}, ${midY + 12})`}>
                                            <rect
                                                x="-68" y="-10" width="136" height="20" rx="4"
                                                fill="rgba(8, 12, 24, 0.9)"
                                                stroke="rgba(168, 85, 247, 0.5)"
                                                strokeWidth="1"
                                            />
                                            <text
                                                fill="#d8b4fe"
                                                fontSize="8.5"
                                                fontFamily="var(--font-mono)"
                                                fontWeight="700"
                                                textAnchor="middle"
                                                dy="3"
                                            >
                                                ⚠ UNOBSERVED CORRIDOR
                                            </text>
                                        </g>
                                    </g>
                                );
                            }

                            // Standard Corridors
                            const midX = (u.x + v.x) / 2;
                            const midY = (u.y + v.y) / 2;
                            const estFlow = Math.round(220 + congScore * 300);
                            const estTime = (road.lengthKm / ((road.baseSpeed * (1 - congScore * 0.45)) || 1) * 60).toFixed(1);

                            return (
                                <g
                                    key={road.id}
                                    opacity={edgeOpacity}
                                    style={{ cursor: 'pointer', transition: 'opacity 0.2s ease' }}
                                    onMouseEnter={() => setHoveredRoadId(road.id)}
                                    onMouseLeave={() => setHoveredRoadId(null)}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setSelectedRoadId(road.id);
                                        setSelectedJunctionId(null);
                                    }}
                                >
                                    {/* Broad Glow Layer */}
                                    <line
                                        x1={u.x} y1={u.y} x2={v.x} y2={v.y}
                                        stroke={baseColor}
                                        strokeWidth={strokeW + (isSelected ? 10 : 6)}
                                        strokeOpacity={isSelected ? 0.35 : 0.12}
                                        strokeLinecap="round"
                                        filter="url(#dt-glow-strong)"
                                    />

                                    {/* Main Road Bed */}
                                    <line
                                        x1={u.x} y1={u.y} x2={v.x} y2={v.y}
                                        stroke="rgba(10, 18, 32, 0.9)"
                                        strokeWidth={strokeW + 2}
                                        strokeLinecap="round"
                                    />

                                    {/* Inner Glowing Traffic Track */}
                                    <line
                                        x1={u.x} y1={u.y} x2={v.x} y2={v.y}
                                        stroke={baseColor}
                                        strokeWidth={isEmergency ? strokeW + 2 : strokeW}
                                        strokeOpacity={isEmergency ? 1.0 : 0.85}
                                        strokeLinecap="round"
                                        strokeDasharray={isEmergency ? '10 5' : 'none'}
                                        filter={isSelected || isEmergency ? 'url(#dt-glow-subtle)' : undefined}
                                    />

                                    {/* Directional Chevrons along road */}
                                    {(() => {
                                        const dx = v.x - u.x;
                                        const dy = v.y - u.y;
                                        const angle = Math.atan2(dy, dx) * (180 / Math.PI);
                                        return (
                                            <g transform={`translate(${midX}, ${midY}) rotate(${angle})`}>
                                                <path
                                                    d="M -6 -3 L 0 0 L -6 3"
                                                    fill="none"
                                                    stroke={baseColor}
                                                    strokeWidth="1.5"
                                                    opacity="0.9"
                                                />
                                            </g>
                                        );
                                    })()}

                                    {/* Road HUD Metrics Pill Label */}
                                    <g transform={`translate(${midX}, ${midY - 14})`}>
                                        <rect
                                            x="-46" y="-8" width="92" height="16" rx="3"
                                            fill="rgba(4, 8, 16, 0.9)"
                                            stroke={isSelected ? 'var(--accent-cyan)' : 'rgba(0, 212, 255, 0.18)'}
                                            strokeWidth="0.8"
                                        />
                                        <text
                                            fill="var(--text-primary)"
                                            fontSize="7.5"
                                            fontFamily="var(--font-mono)"
                                            fontWeight="600"
                                            textAnchor="middle"
                                            dy="3"
                                        >
                                            {estFlow} v/h • {estTime}m
                                        </text>
                                    </g>
                                </g>
                            );
                        })}

                        {/* ============================================================
                            ANIMATED TRAFFIC FLOW PARTICLES
                            ============================================================ */}
                        {particles.map(p => {
                            const pos = getParticlePos(p);
                            if (!pos) return null;
                            if (pos.road.hasUnobservedGap && !showUnobserved) return null;

                            const isConn = isRoadConnectedToActive(pos.road);
                            const pOpacity = activeJunction || activeRoad ? (isConn ? 1.0 : 0.2) : 0.95;

                            return (
                                <g key={p.id} transform={`translate(${pos.x}, ${pos.y})`} opacity={pOpacity}>
                                    <circle
                                        r={p.size}
                                        fill={p.color}
                                        filter="url(#dt-glow-subtle)"
                                    />
                                    {/* Particle Pulse Trail */}
                                    <circle
                                        r={p.size * 2}
                                        fill="none"
                                        stroke={p.color}
                                        strokeWidth="0.75"
                                        opacity="0.4"
                                    />
                                </g>
                            );
                        })}

                        {/* ============================================================
                            INCIDENT RADAR ALERTS
                            ============================================================ */}
                        {incidents.map((_inc, i) => {
                            const targetJunc = junctionMap['JUNC_2'];
                            const angle = (i / Math.max(incidents.length, 1)) * Math.PI * 2;
                            const ix = targetJunc.x + Math.cos(angle) * 38;
                            const iy = targetJunc.y + Math.sin(angle) * 38;

                            return (
                                <g key={`inc_${i}`} transform={`translate(${ix}, ${iy})`}>
                                    <circle r="14" fill="rgba(255, 24, 96, 0.25)" stroke="#ff1860" strokeWidth="1.5" filter="url(#dt-glow-subtle)" />
                                    <circle r="20" fill="none" stroke="#ff1860" strokeWidth="0.8" opacity="0.4" strokeDasharray="3 3" />
                                    <text fill="#ff1860" fontSize="10" fontWeight="900" textAnchor="middle" dy="3.5" fontFamily="var(--font-mono)">!</text>
                                </g>
                            );
                        })}

                        {/* ============================================================
                            JUNCTION NODES (Intelligent Radial HUD)
                            ============================================================ */}
                        {JUNCTIONS.map(junc => {
                            const isHub = junc.id === 'JUNC_5';
                            const isSelected = selectedJunctionId === junc.id;
                            const isHovered = hoveredJunctionId === junc.id;
                            const isConn = isNodeConnectedToActive(junc.id);
                            const isEmergency = isEmergencyJunction(junc.id);

                            const m = getJunctionMetrics(junc.id);
                            const sig = signals[junc.id] || 'GREEN';
                            const sigColor = sig === 'GREEN' ? '#00ff9d' : sig === 'RED' ? '#ff1860' : '#ffb800';

                            const nodeOpacity = activeJunction ? (isConn ? 1.0 : 0.35) : 1.0;
                            const coreRadius = isHub ? 26 : 20;

                            return (
                                <g
                                    key={junc.id}
                                    transform={`translate(${junc.x}, ${junc.y})`}
                                    opacity={nodeOpacity}
                                    style={{ cursor: 'pointer', transition: 'all 0.2s ease' }}
                                    onMouseEnter={() => setHoveredJunctionId(junc.id)}
                                    onMouseLeave={() => setHoveredJunctionId(null)}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setSelectedJunctionId(junc.id);
                                        setSelectedRoadId(null);
                                    }}
                                >
                                    {/* Orbital Hub Aura */}
                                    {isHub && (
                                        <>
                                            <circle r="38" fill="url(#hub-radial-glow)" />
                                            <circle r="34" fill="none" stroke="rgba(139, 0, 255, 0.4)" strokeWidth="1" strokeDasharray="4 4" />
                                        </>
                                    )}

                                    {/* Selection / Emergency Halo */}
                                    {(isSelected || isHovered || isEmergency) && (
                                        <circle
                                            r={coreRadius + 12}
                                            fill="none"
                                            stroke={isEmergency ? 'var(--accent-cyan)' : isSelected ? '#00d4ff' : 'rgba(0, 212, 255, 0.4)'}
                                            strokeWidth="1.5"
                                            strokeDasharray="6 4"
                                            filter="url(#dt-glow-strong)"
                                        />
                                    )}

                                    {/* Outer Status Ring */}
                                    <circle
                                        r={coreRadius + 4}
                                        fill="none"
                                        stroke={m.color}
                                        strokeWidth="1.5"
                                        opacity="0.8"
                                    />

                                    {/* Frosted Core */}
                                    <circle
                                        r={coreRadius}
                                        fill="rgba(6, 12, 24, 0.95)"
                                        stroke={isSelected ? 'var(--accent-cyan)' : 'rgba(0, 212, 255, 0.3)'}
                                        strokeWidth={isSelected ? 2 : 1.2}
                                        filter="url(#dt-glow-subtle)"
                                    />

                                    {/* Signal Phase Indicator */}
                                    <circle
                                        cx={isHub ? 18 : 14}
                                        cy={isHub ? -18 : -14}
                                        r="4.5"
                                        fill={sigColor}
                                        stroke="#020812"
                                        strokeWidth="1.5"
                                        style={{ filter: `drop-shadow(0 0 5px ${sigColor})` }}
                                    />

                                    {/* Junction ID */}
                                    <text
                                        fill={isSelected ? '#00d4ff' : 'var(--text-primary)'}
                                        fontSize={isHub ? '12' : '10.5'}
                                        fontWeight="900"
                                        fontFamily="var(--font-mono)"
                                        textAnchor="middle"
                                        dy="4"
                                    >
                                        {junc.shortLabel}
                                    </text>

                                    {/* Node Label Below */}
                                    <g transform={`translate(0, ${coreRadius + 15})`}>
                                        <rect
                                            x="-48" y="-7" width="96" height="15" rx="3"
                                            fill="rgba(4, 8, 16, 0.85)"
                                            stroke="rgba(0, 212, 255, 0.15)"
                                            strokeWidth="0.6"
                                        />
                                        <text
                                            fill="var(--text-primary)"
                                            fontSize="7.5"
                                            fontFamily="var(--font-sans)"
                                            fontWeight="700"
                                            textAnchor="middle"
                                            dy="4"
                                        >
                                            {junc.label}
                                        </text>
                                    </g>

                                    {/* Flow micro-label */}
                                    <text
                                        fill="var(--accent-cyan)"
                                        fontSize="6.8"
                                        fontFamily="var(--font-mono)"
                                        fontWeight="600"
                                        textAnchor="middle"
                                        dy={coreRadius + 32}
                                    >
                                        {m.flowRate} v/h
                                    </text>
                                </g>
                            );
                        })}
                    </svg>
                </div>

                {/* ============================================================
                    4. FLOATING NODE / CORRIDOR INTELLIGENCE HUD CARD
                    ============================================================ */}
                {inspectedJunction && (
                    <div style={{
                        position: 'absolute',
                        bottom: '16px',
                        left: '16px',
                        zIndex: 10,
                        width: '260px',
                        background: 'rgba(6, 12, 24, 0.95)',
                        backdropFilter: 'blur(12px)',
                        border: '1px solid var(--accent-cyan)',
                        borderRadius: '8px',
                        padding: '0.85rem',
                        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.7), 0 0 16px rgba(0, 212, 255, 0.2)',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', borderBottom: '1px solid rgba(0,212,255,0.15)', paddingBottom: '0.35rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                                <Radio size={12} color="var(--accent-cyan)" />
                                <span style={{ fontSize: '0.72rem', fontWeight: 800, color: '#fff', fontFamily: 'var(--font-display)' }}>
                                    {inspectedJunction.shortLabel} · {inspectedJunction.label}
                                </span>
                            </div>
                            <span className="badge" style={{
                                fontSize: '0.55rem',
                                padding: '1px 5px',
                                background: `${inspectedJunction.metrics.color}20`,
                                color: inspectedJunction.metrics.color,
                                border: `1px solid ${inspectedJunction.metrics.color}50`,
                            }}>
                                {inspectedJunction.metrics.status}
                            </span>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                            <div>
                                <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.55rem' }}>THROUGHPUT FLOW</span>
                                <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{inspectedJunction.metrics.flowRate} veh/h</span>
                            </div>
                            <div>
                                <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.55rem' }}>AVG TRAVEL SPEED</span>
                                <span style={{ color: '#00ff9d', fontWeight: 700 }}>{inspectedJunction.metrics.avgSpeed} km/h</span>
                            </div>
                            <div>
                                <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.55rem' }}>CONGESTION DENSITY</span>
                                <span style={{ color: inspectedJunction.metrics.color, fontWeight: 700 }}>{inspectedJunction.metrics.density}</span>
                            </div>
                            <div>
                                <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.55rem' }}>CORRIDOR TRANSIT</span>
                                <span style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>{inspectedJunction.metrics.travelTime} min</span>
                            </div>
                        </div>

                        <div style={{ marginTop: '0.5rem', paddingTop: '0.4rem', borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'space-between', fontSize: '0.58rem', color: 'var(--text-muted)' }}>
                            <span>INBOUND: {inspectedJunction.metrics.inbound} · OUTBOUND: {inspectedJunction.metrics.outbound}</span>
                            <span style={{ color: 'var(--accent-cyan)' }}>{isDemoMode ? 'DEMO DATA' : 'LIVE'}</span>
                        </div>
                    </div>
                )}

                {/* Corridor Inspection Card */}
                {inspectedRoad && !inspectedJunction && (
                    <div style={{
                        position: 'absolute',
                        bottom: '16px',
                        left: '16px',
                        zIndex: 10,
                        width: '280px',
                        background: 'rgba(6, 12, 24, 0.95)',
                        backdropFilter: 'blur(12px)',
                        border: '1px solid var(--accent-cyan)',
                        borderRadius: '8px',
                        padding: '0.85rem',
                        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.7), 0 0 16px rgba(0, 212, 255, 0.2)',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem', borderBottom: '1px solid rgba(0,212,255,0.15)', paddingBottom: '0.35rem' }}>
                            <span style={{ fontSize: '0.72rem', fontWeight: 800, color: '#fff', fontFamily: 'var(--font-display)' }}>
                                {inspectedRoad.name}
                            </span>
                            <span style={{ fontSize: '0.55rem', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{inspectedRoad.id}</span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                            <div>
                                <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.55rem' }}>LANES & DISTANCE</span>
                                <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{inspectedRoad.lanes} Lanes · {inspectedRoad.lengthKm} km</span>
                            </div>
                            <div>
                                <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.55rem' }}>DESIGN SPEED</span>
                                <span style={{ color: '#00ff9d', fontWeight: 700 }}>{inspectedRoad.baseSpeed} km/h</span>
                            </div>
                        </div>
                        {inspectedRoad.hasUnobservedGap && (
                            <div style={{ marginTop: '0.4rem', fontSize: '0.6rem', color: 'var(--accent-purple)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <AlertTriangle size={11} /> Unmonitored blind zone sector between monitored hops
                            </div>
                        )}
                    </div>
                )}

                {/* ============================================================
                    INSET CITY OVERVIEW RADAR MINI-MAP
                    ============================================================ */}
                <div style={{
                    position: 'absolute',
                    bottom: '14px',
                    right: '14px',
                    zIndex: 6,
                    width: '120px',
                    height: '90px',
                    background: 'rgba(4, 8, 16, 0.92)',
                    borderRadius: '6px',
                    border: '1px solid rgba(0, 212, 255, 0.3)',
                    padding: '4px',
                    overflow: 'hidden',
                    boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
                }}>
                    <div style={{ fontSize: '0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '3px', marginBottom: '2px' }}>
                        <Compass size={8} /> CITY OVERVIEW
                    </div>
                    <svg width="100%" height="70" viewBox="0 0 620 440">
                        {visibleRoads.map(r => {
                            const u = junctionMap[r.from];
                            const v = junctionMap[r.to];
                            if (!u || !v) return null;
                            return (
                                <line
                                    key={`mini_${r.id}`}
                                    x1={u.x} y1={u.y} x2={v.x} y2={v.y}
                                    stroke="rgba(0, 212, 255, 0.4)"
                                    strokeWidth="6"
                                />
                            );
                        })}
                        {JUNCTIONS.map(j => (
                            <circle
                                key={`mini_j_${j.id}`}
                                cx={j.x} cy={j.y}
                                r={j.id === 'JUNC_5' ? 22 : 16}
                                fill={selectedJunctionId === j.id ? '#00d4ff' : 'rgba(0, 255, 157, 0.8)'}
                            />
                        ))}
                    </svg>
                </div>
            </div>

            {/* ============================================================
                5. TIME PLAYBACK SCRUBBER BAR
                ============================================================ */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: 'rgba(4, 7, 14, 0.85)',
                padding: '0.5rem 0.85rem',
                borderRadius: '8px',
                border: '1px solid rgba(0, 212, 255, 0.15)',
                flexWrap: 'wrap',
                gap: '0.6rem',
            }}>
                {/* Play/Pause & Speed */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        style={{
                            background: isPlaying ? 'rgba(0,212,255,0.15)' : 'rgba(255,24,96,0.15)',
                            border: `1px solid ${isPlaying ? 'var(--accent-cyan)' : 'var(--accent-rose)'}`,
                            color: isPlaying ? 'var(--accent-cyan)' : 'var(--accent-rose)',
                            borderRadius: '5px',
                            padding: '4px 10px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '0.62rem',
                            fontWeight: 700,
                            fontFamily: 'var(--font-mono)',
                        }}
                    >
                        {isPlaying ? <Pause size={11} /> : <Play size={11} />}
                        {isPlaying ? 'PAUSE' : 'PLAY'}
                    </button>

                    {[1, 5, 10].map(spd => (
                        <button
                            key={spd}
                            onClick={() => setPlaybackSpeed(spd)}
                            style={{
                                background: playbackSpeed === spd ? 'rgba(0,212,255,0.2)' : 'transparent',
                                border: `1px solid ${playbackSpeed === spd ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.1)'}`,
                                color: playbackSpeed === spd ? 'var(--accent-cyan)' : 'var(--text-muted)',
                                borderRadius: '4px',
                                padding: '2px 6px',
                                fontSize: '0.58rem',
                                fontFamily: 'var(--font-mono)',
                                cursor: 'pointer',
                            }}
                        >
                            {spd}×
                        </button>
                    ))}
                </div>

                {/* Timeline Stepper */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flex: 1, justifyContent: 'center' }}>
                    <Clock size={12} color="var(--text-muted)" />
                    <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginRight: '6px' }}>
                        TIME REPLAY (SIMULATED):
                    </span>
                    {TIMELINE_STEPS.map((t, idx) => (
                        <button
                            key={t}
                            onClick={() => setPlaybackTimeIdx(idx)}
                            style={{
                                background: playbackTimeIdx === idx ? 'rgba(0,212,255,0.25)' : 'rgba(8,14,26,0.6)',
                                border: `1px solid ${playbackTimeIdx === idx ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.06)'}`,
                                color: playbackTimeIdx === idx ? '#fff' : 'var(--text-secondary)',
                                borderRadius: '4px',
                                padding: '2px 8px',
                                fontSize: '0.6rem',
                                fontFamily: 'var(--font-mono)',
                                fontWeight: playbackTimeIdx === idx ? 700 : 400,
                                cursor: 'pointer',
                            }}
                        >
                            {t}
                        </button>
                    ))}
                </div>

                <span className="demo-tag" style={{ fontSize: '0.55rem' }}>
                    DEMO DATA
                </span>
            </div>

            {/* ============================================================
                6. NETWORK METRICS & CORRIDOR ANALYSIS GRID
                ============================================================ */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.65rem' }}>
                {[
                    { label: 'TOTAL NETWORK FLOW', value: '1,380 veh/h', change: '+8.2%', icon: <Activity size={14} color="#00d4ff" />, color: '#00d4ff' },
                    { label: 'AVG NETWORK SPEED', value: '46.4 km/h', change: '-3.1%', icon: <FastForward size={14} color="#00ff9d" />, color: '#00ff9d' },
                    { label: 'NETWORK CONGESTION DENSITY', value: '0.38', change: 'NOMINAL', icon: <BarChart3 size={14} color="#ffb800" />, color: '#ffb800' },
                    { label: 'AVG NETWORK TRAVEL TIME', value: '4.1 min', change: 'OPTIMAL', icon: <Clock size={14} color="#d8b4fe" />, color: '#d8b4fe' },
                ].map(card => (
                    <div
                        key={card.label}
                        style={{
                            background: 'var(--bg-inset)',
                            border: '1px solid rgba(0, 212, 255, 0.12)',
                            borderRadius: '8px',
                            padding: '0.65rem 0.85rem',
                        }}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                            <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{card.label}</span>
                            {card.icon}
                        </div>
                        <div style={{ fontSize: '1.05rem', fontWeight: 800, color: card.color, fontFamily: 'var(--font-mono)' }}>
                            {card.value}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2px', fontSize: '0.55rem', color: 'var(--text-secondary)' }}>
                            <span>Trend: {card.change}</span>
                            <span style={{ color: 'var(--text-muted)' }}>{isDemoMode ? 'DEMO DATA' : 'LIVE'}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* ============================================================
                7. CORRIDOR ANALYSIS TABLE
                ============================================================ */}
            <div style={{
                background: 'var(--bg-inset)',
                borderRadius: '8px',
                border: '1px solid rgba(0, 212, 255, 0.15)',
                overflow: 'hidden',
            }}>
                <div style={{
                    padding: '0.5rem 0.85rem',
                    borderBottom: '1px solid rgba(0, 212, 255, 0.12)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                }}>
                    <span style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: '5px' }}>
                        <ArrowUpRight size={13} color="var(--accent-cyan)" />
                        ACTIVE CORRIDOR FLOW & SPATIO-TEMPORAL TRAVEL TIME DYNAMICS
                    </span>
                    <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)' }}>Click corridor to highlight on graph</span>
                </div>

                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
                        <thead>
                            <tr style={{ background: 'rgba(255,255,255,0.02)', color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                <th style={{ padding: '6px 12px', textAlign: 'left' }}>FROM → TO CORRIDOR</th>
                                <th style={{ padding: '6px 12px', textAlign: 'center' }}>LANES</th>
                                <th style={{ padding: '6px 12px', textAlign: 'center' }}>FLOW RATE</th>
                                <th style={{ padding: '6px 12px', textAlign: 'center' }}>AVG SPEED</th>
                                <th style={{ padding: '6px 12px', textAlign: 'center' }}>EST. TRAVEL TIME</th>
                                <th style={{ padding: '6px 12px', textAlign: 'right' }}>CONGESTION</th>
                            </tr>
                        </thead>
                        <tbody>
                            {visibleRoads.map(road => {
                                const fromNode = junctionMap[road.from];
                                const toNode = junctionMap[road.to];
                                const cong = snapshot.segment_states[road.segKey]?.congestion_score ?? (road.hasUnobservedGap ? 0.3 : 0.42);
                                const color = road.hasUnobservedGap ? '#a855f7' : congestionColor(cong);
                                const isSelected = selectedRoadId === road.id;

                                return (
                                    <tr
                                        key={road.id}
                                        onClick={() => {
                                            setSelectedRoadId(isSelected ? null : road.id);
                                            setSelectedJunctionId(null);
                                        }}
                                        style={{
                                            borderBottom: '1px solid rgba(255,255,255,0.03)',
                                            background: isSelected ? 'rgba(0, 212, 255, 0.1)' : 'transparent',
                                            cursor: 'pointer',
                                            transition: 'background 0.15s ease',
                                        }}
                                    >
                                        <td style={{ padding: '6px 12px' }}>
                                            <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{fromNode?.shortLabel}</span>
                                            <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>→</span>
                                            <span style={{ color: '#00ff9d', fontWeight: 700 }}>{toNode?.shortLabel}</span>
                                            <span style={{ color: 'var(--text-secondary)', marginLeft: '8px', fontSize: '0.58rem' }}>({road.name})</span>
                                        </td>
                                        <td style={{ padding: '6px 12px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                                            {road.lanes}
                                        </td>
                                        <td style={{ padding: '6px 12px', textAlign: 'center', color: 'var(--accent-cyan)', fontWeight: 700 }}>
                                            {road.hasUnobservedGap ? '180 v/h' : `${Math.round(240 + cong * 280)} v/h`}
                                        </td>
                                        <td style={{ padding: '6px 12px', textAlign: 'center', color: '#00ff9d' }}>
                                            {road.hasUnobservedGap ? '35 km/h' : `${Math.round(road.baseSpeed * (1 - cong * 0.4))} km/h`}
                                        </td>
                                        <td style={{ padding: '6px 12px', textAlign: 'center', color: 'var(--accent-amber)', fontWeight: 700 }}>
                                            {(road.lengthKm / (road.baseSpeed * (1 - cong * 0.4) || 1) * 60).toFixed(1)} min
                                        </td>
                                        <td style={{ padding: '6px 12px', textAlign: 'right' }}>
                                            <span style={{
                                                fontSize: '0.55rem',
                                                padding: '2px 6px',
                                                borderRadius: '3px',
                                                background: `${color}20`,
                                                color: color,
                                                border: `1px solid ${color}60`,
                                                fontWeight: 700,
                                            }}>
                                                {road.hasUnobservedGap ? 'UNOBSERVED' : congestionLabel(cong)}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* ============================================================
                8. MODULE 2 → MODULE 3 PIPELINE HANDOFF BANNER
                ============================================================ */}
            <div style={{
                background: 'rgba(8, 14, 26, 0.8)',
                borderRadius: '6px',
                border: '1px solid rgba(0, 212, 255, 0.15)',
                padding: '0.5rem 0.85rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '0.62rem',
                fontFamily: 'var(--font-mono)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>MODULE 2 ReID Journeys</span>
                    <ChevronRight size={12} color="var(--text-muted)" />
                    <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>Dynamic Edge Flows</span>
                    <ChevronRight size={12} color="var(--text-muted)" />
                    <span style={{ color: 'var(--accent-purple)', fontWeight: 700 }}>Spatio-Temporal Graph</span>
                    <ChevronRight size={12} color="var(--text-muted)" />
                    <span style={{ color: 'var(--text-muted)', fontWeight: 700 }}>MODULE 3 ST-GNN</span>
                </div>
                <span className="badge badge-amber" style={{ fontSize: '0.55rem' }}>
                    ST-GNN MODEL NOT CONNECTED (DEMO MODE)
                </span>
            </div>
        </div>
    );
};
