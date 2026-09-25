import React, { useState, useEffect } from 'react';
import { Search, Eye, Fingerprint, Route, FileCheck } from 'lucide-react';
import { fetchReIDJourney, fetchGlobalVehicles } from '../services/api';
import { VehicleJourney, GlobalVehicleSummary } from '../types/dashboard';
import { SevenSignalPanel } from './SevenSignalPanel';
import { OcrEvidencePanel } from './OcrEvidencePanel';

const VEHICLE_TYPES: Record<string, { emoji: string; color: string }> = {
    car: { emoji: '🚗', color: 'var(--accent-cyan)' },
    truck: { emoji: '🚛', color: 'var(--accent-amber)' },
    bus: { emoji: '🚌', color: 'var(--accent-green)' },
    motorcycle: { emoji: '🏍️', color: 'var(--accent-rose)' },
    auto: { emoji: '🛺', color: 'var(--accent-purple)' },
};

interface VehicleInspectorProps {
    initialVehicleId?: string;
    onSelectJourney?: (journey: VehicleJourney) => void;
    isDemoMode?: boolean;
}

export const VehicleInspector: React.FC<VehicleInspectorProps> = ({
    initialVehicleId = 'VEH_101',
    onSelectJourney,
    isDemoMode = false,
}) => {
    const [searchTerm, setSearchTerm] = useState(initialVehicleId);
    const [searchFocused, setSearchFocused] = useState(false);
    const [activeTab, setActiveTab] = useState<'TRANSITIONS' | 'SEVEN_SIGNAL' | 'OCR_FUSION'>('TRANSITIONS');
    const [journeyData, setJourneyData] = useState<VehicleJourney | null>(null);
    const [allVehicles, setAllVehicles] = useState<GlobalVehicleSummary[]>([]);
    const [selectedTransitionIdx, setSelectedTransitionIdx] = useState<number>(0);

    // Sync with parent props
    useEffect(() => {
        if (initialVehicleId) {
            setSearchTerm(initialVehicleId);
        }
    }, [initialVehicleId]);

    // Load available global vehicles
    useEffect(() => {
        let isMounted = true;
        async function loadVehicles() {
            const res = await fetchGlobalVehicles();
            if (isMounted && res && res.vehicles && res.vehicles.length > 0) {
                setAllVehicles(res.vehicles);
            }
        }
        loadVehicles();
        return () => { isMounted = false; };
    }, []);

    // Load active journey
    useEffect(() => {
        let isMounted = true;
        async function loadJourney() {
            if (!searchTerm) return;
            const data = await fetchReIDJourney(searchTerm);
            if (isMounted && data) {
                setJourneyData(data);
                if (onSelectJourney) {
                    onSelectJourney(data);
                }
            }
        }
        loadJourney();
        return () => { isMounted = false; };
    }, [searchTerm]);

    if (!journeyData) {
        return (
            <div className="glass-panel" style={{ padding: '1.5rem', textAlign: 'center', marginBottom: '1rem' }}>
                <Eye size={28} color="var(--accent-cyan)" style={{ opacity: 0.5, margin: '0 auto 0.5rem' }} />
                <h3 style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>Loading Vehicle Intelligence...</h3>
            </div>
        );
    }

    const isDemo = isDemoMode || (journeyData as any)._demo;
    const vehType = VEHICLE_TYPES[journeyData.vehicle_type || 'car'] || VEHICLE_TYPES.car;
    const confPct = Math.round((journeyData.overall_confidence ?? 0.94) * 100);
    const segments = journeyData.segments || [];
    const activeSegment = segments[selectedTransitionIdx] || segments[0];

    // Compute first & last observed times
    const firstSeen = journeyData.first_seen
        ? new Date(journeyData.first_seen * 1000).toLocaleTimeString()
        : '10:14:20 AM';
    const lastSeen = journeyData.last_seen
        ? new Date(journeyData.last_seen * 1000).toLocaleTimeString()
        : '10:17:10 AM';

    const getPlateBadge = (status?: string | null) => {
        switch (status) {
            case 'CONFIRMED':
                return { label: 'CONFIRMED', color: 'var(--accent-green)', bg: 'rgba(0,255,157,0.12)', border: 'rgba(0,255,157,0.3)' };
            case 'PENDING':
                return { label: 'PENDING', color: 'var(--accent-amber)', bg: 'rgba(255,184,0,0.12)', border: 'rgba(255,184,0,0.3)' };
            case 'UNKNOWN':
            default:
                return { label: 'UNKNOWN', color: 'var(--accent-rose)', bg: 'rgba(255,24,96,0.12)', border: 'rgba(255,24,96,0.3)' };
        }
    };

    const plateBadge = getPlateBadge(activeSegment?.plate_status || 'CONFIRMED');

    return (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Eye size={18} color="var(--accent-cyan)" />
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <h2 style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: '0.72rem',
                                fontWeight: 700,
                                color: 'var(--text-primary)',
                                letterSpacing: '0.1em',
                            }}>
                                CROSS-CAMERA RE-ID & VEHICLE INTELLIGENCE INSPECTOR
                            </h2>
                            {isDemo && (
                                <span className="demo-tag" style={{ fontSize: '0.55rem', padding: '1px 5px', borderRadius: '3px' }}>
                                    DEMO DATA
                                </span>
                            )}
                        </div>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Spatial-Temporal Kinematics · 7-Signal ReID Fusion · Multi-Frame OCR Consensus
                        </p>
                    </div>
                </div>

                {/* Search */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    background: 'rgba(4,7,14,0.85)',
                    border: `1px solid ${searchFocused ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                    borderRadius: '8px',
                    padding: '0.4rem 0.75rem',
                    boxShadow: searchFocused ? '0 0 0 3px rgba(0,212,255,0.1)' : 'none',
                    transition: 'all 0.2s ease',
                }}>
                    <Search size={13} color={searchFocused ? 'var(--accent-cyan)' : 'var(--text-muted)'} style={{ marginRight: '0.4rem' }} />
                    <input
                        id="vehicle-search"
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onFocus={() => setSearchFocused(true)}
                        onBlur={() => setSearchFocused(false)}
                        placeholder="Search Plate or VEH_xxx..."
                        style={{
                            background: 'none',
                            border: 'none',
                            color: 'var(--text-primary)',
                            fontSize: '0.78rem',
                            fontFamily: 'var(--font-mono)',
                            outline: 'none',
                            width: '200px',
                        }}
                    />
                </div>
            </div>

            {/* Quick Vehicle Select Badges */}
            {allVehicles.length > 0 && (
                <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1rem', overflowX: 'auto', paddingBottom: '2px' }}>
                    <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', alignSelf: 'center', marginRight: '4px' }}>
                        ACTIVE TARGETS:
                    </span>
                    {allVehicles.slice(0, 6).map((v) => {
                        const isCurr = (v.global_vehicle_id === journeyData.global_vehicle_id) || (v.plate_number === journeyData.plate_number);
                        return (
                            <button
                                key={v.global_vehicle_id || v.plate_number}
                                onClick={() => setSearchTerm(v.global_vehicle_id || v.plate_number)}
                                style={{
                                    background: isCurr ? 'rgba(0,212,255,0.18)' : 'rgba(4,7,14,0.6)',
                                    border: `1px solid ${isCurr ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                                    color: isCurr ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                                    fontSize: '0.68rem',
                                    fontFamily: 'var(--font-mono)',
                                    padding: '2px 8px',
                                    borderRadius: '4px',
                                    cursor: 'pointer',
                                    transition: 'all 0.15s ease',
                                    fontWeight: isCurr ? 700 : 500,
                                }}
                            >
                                {v.global_vehicle_id}: {v.plate_number || 'UNKNOWN'}
                            </button>
                        );
                    })}
                </div>
            )}

            {/* Vehicle Identity Header Card */}
            <div style={{
                background: 'var(--bg-inset)',
                border: '1px solid rgba(0,212,255,0.15)',
                borderRadius: '8px',
                padding: '1rem',
                marginBottom: '1rem',
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    paddingBottom: '0.75rem',
                    borderBottom: '1px solid var(--border-color)',
                    marginBottom: '0.75rem',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div style={{
                            width: '46px',
                            height: '46px',
                            borderRadius: '10px',
                            background: `${vehType.color}15`,
                            border: `1px solid ${vehType.color}40`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '1.4rem',
                        }}>
                            {vehType.emoji}
                        </div>

                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                                <div>
                                    <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                        GLOBAL VEHICLE ID
                                    </span>
                                    <div style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                                        {journeyData.global_vehicle_id}
                                    </div>
                                </div>

                                <div style={{ width: '1px', height: '28px', background: 'var(--border-color)' }} />

                                <div>
                                    <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                        LICENSE PLATE
                                    </span>
                                    <div style={{
                                        fontSize: '0.95rem',
                                        fontWeight: 800,
                                        color: 'var(--accent-green)',
                                        fontFamily: 'var(--font-mono)',
                                        background: 'rgba(0,255,157,0.08)',
                                        padding: '1px 8px',
                                        borderRadius: '4px',
                                        border: '1px solid rgba(0,255,157,0.25)',
                                    }}>
                                        {journeyData.plate_number || 'UNKNOWN'}
                                    </div>
                                </div>

                                <div style={{ width: '1px', height: '28px', background: 'var(--border-color)' }} />

                                <div>
                                    <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                        PLATE STATUS
                                    </span>
                                    <div>
                                        <span style={{
                                            fontSize: '0.68rem',
                                            fontWeight: 700,
                                            fontFamily: 'var(--font-mono)',
                                            color: plateBadge.color,
                                            background: plateBadge.bg,
                                            border: `1px solid ${plateBadge.border}`,
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                        }}>
                                            {plateBadge.label}
                                        </span>
                                    </div>
                                </div>

                                <div style={{ width: '1px', height: '28px', background: 'var(--border-color)' }} />

                                <div>
                                    <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                        TYPE
                                    </span>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 700, color: vehType.color, textTransform: 'uppercase' }}>
                                        {journeyData.vehicle_type || 'car'}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.58rem', color: 'var(--text-muted)', marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                            GLOBAL REID CONFIDENCE
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <div className="progress-bar-track" style={{ width: '60px' }}>
                                <div
                                    className="progress-bar-fill"
                                    style={{
                                        width: `${confPct}%`,
                                        background: confPct > 85
                                            ? 'linear-gradient(90deg, #00d4ff, var(--accent-green))'
                                            : 'linear-gradient(90deg, #ffb800, var(--accent-amber))',
                                    }}
                                />
                            </div>
                            <span className={`badge ${confPct > 85 ? 'badge-green' : 'badge-amber'}`}>
                                <Fingerprint size={10} /> {confPct}%
                            </span>
                        </div>
                    </div>
                </div>

                {/* Journey Summary Stats */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: '0.75rem',
                    fontSize: '0.7rem',
                    fontFamily: 'var(--font-mono)',
                }}>
                    <div>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>FIRST OBSERVED: </span>
                        <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{firstSeen}</div>
                    </div>
                    <div>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>LAST OBSERVED: </span>
                        <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{lastSeen}</div>
                    </div>
                    <div>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>CAMERA HOPS: </span>
                        <div style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{segments.length} Cameras</div>
                    </div>
                    <div>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>UNOBSERVED GAPS: </span>
                        <div style={{ color: segments.some(s => s.has_unobserved_gap) ? 'var(--accent-amber)' : 'var(--accent-green)', fontWeight: 700 }}>
                            {segments.filter(s => s.has_unobserved_gap).length} Gaps
                        </div>
                    </div>
                </div>
            </div>

            {/* Sub-view Navigation */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
                <button
                    onClick={() => setActiveTab('TRANSITIONS')}
                    style={{
                        background: activeTab === 'TRANSITIONS' ? 'rgba(0,212,255,0.15)' : 'transparent',
                        border: `1px solid ${activeTab === 'TRANSITIONS' ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                        color: activeTab === 'TRANSITIONS' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        padding: '4px 12px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                    }}
                >
                    <Route size={13} />
                    <span>CAMERA TRANSITIONS ({segments.length})</span>
                </button>

                <button
                    onClick={() => setActiveTab('SEVEN_SIGNAL')}
                    style={{
                        background: activeTab === 'SEVEN_SIGNAL' ? 'rgba(0,212,255,0.15)' : 'transparent',
                        border: `1px solid ${activeTab === 'SEVEN_SIGNAL' ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                        color: activeTab === 'SEVEN_SIGNAL' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        padding: '4px 12px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                    }}
                >
                    <Fingerprint size={13} />
                    <span>SEVEN-SIGNAL REID BREAKDOWN</span>
                </button>

                <button
                    onClick={() => setActiveTab('OCR_FUSION')}
                    style={{
                        background: activeTab === 'OCR_FUSION' ? 'rgba(0,212,255,0.15)' : 'transparent',
                        border: `1px solid ${activeTab === 'OCR_FUSION' ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                        color: activeTab === 'OCR_FUSION' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        padding: '4px 12px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                    }}
                >
                    <FileCheck size={13} />
                    <span>TEMPORAL OCR FUSION</span>
                </button>
            </div>

            {/* Content */}
            {activeTab === 'TRANSITIONS' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {segments.map((seg, idx) => {
                        const isSelected = selectedTransitionIdx === idx;
                        const timeStr = seg.timestamp ? new Date(seg.timestamp * 1000).toLocaleTimeString() : '10:14:20 AM';
                        const score = Math.round((seg.transition_score ?? 0.9) * 100);

                        return (
                            <div
                                key={idx}
                                onClick={() => setSelectedTransitionIdx(idx)}
                                style={{
                                    background: isSelected ? 'rgba(0,212,255,0.08)' : 'var(--bg-inset)',
                                    border: `1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                                    borderRadius: '6px',
                                    padding: '0.65rem 0.85rem',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    cursor: 'pointer',
                                    transition: 'all 0.15s ease',
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                    <div style={{
                                        width: '24px',
                                        height: '24px',
                                        borderRadius: '50%',
                                        background: isSelected ? 'var(--accent-cyan)' : 'rgba(0,212,255,0.2)',
                                        color: isSelected ? '#04070e' : 'var(--accent-cyan)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        fontFamily: 'var(--font-mono)',
                                        fontWeight: 800,
                                        fontSize: '0.75rem',
                                    }}>
                                        {idx + 1}
                                    </div>

                                    <div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                                                {seg.camera_id}
                                            </span>
                                            <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                                                [{seg.track_id}]
                                            </span>
                                            {seg.has_unobserved_gap && (
                                                <span style={{
                                                    fontSize: '0.6rem',
                                                    fontWeight: 700,
                                                    color: 'var(--accent-amber)',
                                                    background: 'rgba(255,184,0,0.12)',
                                                    padding: '1px 5px',
                                                    borderRadius: '3px',
                                                    border: '1px solid rgba(255,184,0,0.3)',
                                                }}>
                                                    [UNOBSERVED SEGMENT]
                                                </span>
                                            )}
                                        </div>
                                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                                            Plate: <strong style={{ color: 'var(--accent-green)' }}>{seg.plate_number || 'UNKNOWN'}</strong> ({seg.plate_status || 'CONFIRMED'})
                                        </div>
                                    </div>
                                </div>

                                <div style={{ textAlign: 'right' }}>
                                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                                        {score}% Match
                                    </div>
                                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                                        {timeStr}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {activeTab === 'SEVEN_SIGNAL' && (
                <SevenSignalPanel
                    breakdown={activeSegment?.transition_breakdown}
                    fromCamera={selectedTransitionIdx > 0 ? segments[selectedTransitionIdx - 1]?.camera_id : 'ORIGIN'}
                    toCamera={activeSegment?.camera_id || 'CAM_A_EAST'}
                    isDemoMode={isDemo}
                />
            )}

            {activeTab === 'OCR_FUSION' && (
                <OcrEvidencePanel
                    plateNumber={journeyData.plate_number || undefined}
                    plateStatus={activeSegment?.plate_status || undefined}
                    plateConfidence={activeSegment?.plate_confidence || undefined}
                    isDemoMode={isDemo}
                />
            )}
        </div>
    );
};
