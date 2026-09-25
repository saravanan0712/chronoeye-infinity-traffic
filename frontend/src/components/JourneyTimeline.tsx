import React from 'react';
import { Eye, Clock, AlertTriangle, Route } from 'lucide-react';
import { VehicleJourney } from '../types/dashboard';

interface JourneyTimelineProps {
    journey: VehicleJourney | null;
    isDemoMode: boolean;
    onSelectSegment?: (segIndex: number) => void;
    selectedSegmentIndex?: number;
}

const formatTime = (ts: number) => {
    if (!ts) return '--:--:--';
    return new Date(ts * 1000).toLocaleTimeString();
};

export const JourneyTimeline: React.FC<JourneyTimelineProps> = ({
    journey, isDemoMode, onSelectSegment, selectedSegmentIndex,
}) => {
    if (!journey || journey.segments.length === 0) {
        return (
            <div className="glass-panel" style={{ padding: '1rem' }}>
                <div className="evidence-unavailable">
                    <Eye size={24} color="var(--text-muted)" />
                    <span>NO JOURNEY SELECTED</span>
                    <span style={{ fontSize: '0.6rem' }}>Select a global vehicle to view its observed journey</span>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-panel" style={{ padding: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Route size={16} color="var(--accent-cyan)" />
                    <h2 style={{
                        fontFamily: 'var(--font-display)', fontSize: '0.68rem', fontWeight: 700,
                        color: 'var(--text-primary)', letterSpacing: '0.1em',
                    }}>
                        OBSERVED JOURNEY TIMELINE
                    </h2>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="identity-label identity-journey">{journey.journey_id}</span>
                    <span className="identity-label identity-vehicle">{journey.global_vehicle_id}</span>
                    {isDemoMode && <span className="demo-mode-badge demo" style={{ fontSize: '0.5rem', padding: '1px 6px' }}>DEMO DATA</span>}
                </div>
            </div>

            {/* Timeline */}
            <div style={{ display: 'flex', flexDirection: 'column', paddingLeft: '0.5rem' }}>
                {journey.segments.map((seg, idx) => {
                    const isSelected = selectedSegmentIndex === idx;
                    const isFirst = idx === 0;
                    const plateStatus = seg.plate_status || 'UNKNOWN';

                    return (
                        <React.Fragment key={seg.segment_id}>
                            {/* Unobserved Gap (before this segment) */}
                            {seg.has_unobserved_gap && !isFirst && (
                                <div style={{ display: 'flex', alignItems: 'stretch', marginLeft: '5px' }}>
                                    <div style={{ width: '2px', minHeight: '40px' }} className="timeline-line timeline-line-gap" />
                                    <div className="unobserved-gap" style={{ marginLeft: '1rem', marginBottom: '0.35rem', flex: 1 }}>
                                        <AlertTriangle size={12} />
                                        UNOBSERVED SEGMENT
                                    </div>
                                </div>
                            )}

                            {/* Transition info (between segments) */}
                            {!isFirst && !seg.has_unobserved_gap && (
                                <div style={{ display: 'flex', alignItems: 'stretch', marginLeft: '5px' }}>
                                    <div style={{ width: '2px', minHeight: '30px' }} className="timeline-line timeline-line-observed" />
                                    <div style={{
                                        marginLeft: '1rem', fontSize: '0.58rem', color: 'var(--text-muted)',
                                        display: 'flex', alignItems: 'center', gap: '0.5rem', paddingBottom: '0.2rem',
                                    }}>
                                        {seg.transition_decision && (
                                             <span className={`decision-badge decision-${seg.transition_decision}`}>
                                                {seg.transition_decision.replace(/_/g, ' ')}
                                            </span>
                                        )}
                                        {seg.transition_score != null && (
                                            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                                                Score: {seg.transition_score.toFixed(2)}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Observation Node */}
                            <div
                                onClick={() => onSelectSegment?.(idx)}
                                style={{
                                    display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
                                    cursor: 'pointer',
                                    padding: '0.55rem 0.65rem', borderRadius: '8px', marginBottom: '0.2rem',
                                    background: isSelected ? 'rgba(0,212,255,0.08)' : 'transparent',
                                    border: isSelected ? '1px solid rgba(0,212,255,0.3)' : '1px solid transparent',
                                    transition: 'all 0.15s',
                                }}
                            >
                                {/* Timeline dot */}
                                <div className="timeline-node" style={{
                                    background: 'var(--accent-cyan)',
                                    boxShadow: '0 0 8px rgba(0,212,255,0.5)',
                                    marginTop: '2px',
                                }} />

                                {/* Content */}
                                <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                                        <span style={{ fontSize: '0.72rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                                            {seg.camera_id}
                                        </span>
                                        <span className="identity-label identity-track">{seg.track_id}</span>
                                    </div>

                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.62rem', color: 'var(--text-secondary)' }}>
                                            <Clock size={10} />
                                            {formatTime(seg.timestamp)}
                                        </div>

                                        <div style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                                            <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)' }}>PLATE:</span>
                                            {seg.plate_number && seg.plate_number !== 'UNKNOWN' ? (
                                                <span style={{
                                                    fontSize: '0.68rem', fontWeight: 700, fontFamily: 'var(--font-mono)',
                                                    color: plateStatus === 'CONFIRMED' ? 'var(--accent-green)' : plateStatus === 'PENDING' ? 'var(--accent-amber)' : 'var(--text-muted)',
                                                }}>
                                                    {seg.plate_number}
                                                </span>
                                            ) : (
                                                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                                    —
                                                </span>
                                            )}
                                            <span className={`decision-badge plate-${plateStatus}`} style={{ fontSize: '0.48rem', padding: '1px 4px' }}>
                                                {plateStatus}
                                            </span>
                                        </div>

                                        <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                                            {seg.speed_estimate.toFixed(0)} km/h · {seg.direction}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </React.Fragment>
                    );
                })}
            </div>
        </div>
    );
};
