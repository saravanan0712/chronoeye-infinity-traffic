import React from 'react';
import { Network, ArrowRight, Share2, Layers, Cpu, Database, CheckCircle2 } from 'lucide-react';
import { VehicleJourney } from '../types/dashboard';

interface Module3HandoffProps {
    journeys?: VehicleJourney[];
    activeVehiclesCount?: number;
    isDemoMode?: boolean;
}

export const Module3Handoff: React.FC<Module3HandoffProps> = ({
    journeys = [],
    activeVehiclesCount = 42,
    isDemoMode = false,
}) => {
    const isDemo = isDemoMode;
    const journeyCount = journeys.length > 0 ? journeys.length : 18;

    // OD Flow pairs derived from journeys
    const odFlows = [
        { origin: 'CAM_A_EAST', destination: 'CAM_B_WEST', count: 14, avgTravelTime: '82s', flowRate: '420 veh/h' },
        { origin: 'CAM_B_WEST', destination: 'CAM_C_NORTH', count: 9, avgTravelTime: '95s', flowRate: '310 veh/h' },
        { origin: 'CAM_C_NORTH', destination: 'CAM_D_SOUTH', count: 12, avgTravelTime: '110s', flowRate: '380 veh/h' },
        { origin: 'CAM_A_EAST', destination: 'CAM_D_SOUTH', count: 6, avgTravelTime: '190s', flowRate: '190 veh/h' },
    ];

    return (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <Network size={18} color="var(--accent-cyan)" />
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <h2 style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: '0.75rem',
                                fontWeight: 700,
                                color: 'var(--text-primary)',
                                letterSpacing: '0.1em',
                            }}>
                                MODULE 3 TEMPORAL TRAFFIC GRAPH HANDOFF
                            </h2>
                            {isDemo && (
                                <span className="demo-tag" style={{ fontSize: '0.55rem', padding: '1px 5px', borderRadius: '3px' }}>
                                    DEMO DATA
                                </span>
                            )}
                        </div>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Cross-Camera ReID Journeys → Dynamic Edge Weights & Spatio-Temporal Graph State
                        </p>
                    </div>
                </div>

                {/* Integration Status Badge */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    background: 'rgba(0,255,157,0.12)',
                    border: '1px solid rgba(0,255,157,0.3)',
                    color: 'var(--accent-green)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.72rem',
                    fontWeight: 700,
                }}>
                    <CheckCircle2 size={13} />
                    <span>GRAPH PIPELINE ACTIVE</span>
                </div>
            </div>

            {/* Architecture Pipeline Flow Diagram */}
            <div style={{
                background: 'var(--bg-inset)',
                borderRadius: '8px',
                border: '1px solid rgba(0,212,255,0.15)',
                padding: '1rem',
                marginBottom: '1rem',
                display: 'grid',
                gridTemplateColumns: '1fr auto 1fr auto 1fr',
                alignItems: 'center',
                gap: '0.75rem',
            }}>
                {/* Stage 1 */}
                <div style={{
                    background: 'rgba(0,212,255,0.06)',
                    border: '1px solid rgba(0,212,255,0.25)',
                    borderRadius: '6px',
                    padding: '0.75rem',
                    textAlign: 'center',
                }}>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '4px' }}>
                        <Database size={16} color="var(--accent-cyan)" />
                    </div>
                    <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>MODULE 2 OUTPUT</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
                        ReID Journeys
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', marginTop: '3px' }}>
                        {journeyCount} Journeys · {activeVehiclesCount} Tracks
                    </div>
                </div>

                {/* Arrow */}
                <div style={{ color: 'var(--accent-cyan)', display: 'flex', justifyContent: 'center' }}>
                    <ArrowRight size={18} />
                </div>

                {/* Stage 2 */}
                <div style={{
                    background: 'rgba(0,255,157,0.06)',
                    border: '1px solid rgba(0,255,157,0.25)',
                    borderRadius: '6px',
                    padding: '0.75rem',
                    textAlign: 'center',
                }}>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '4px' }}>
                        <Share2 size={16} color="var(--accent-green)" />
                    </div>
                    <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>DYNAMIC TOPOLOGY</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
                        Edge Flow Matrix
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--accent-green)', fontFamily: 'var(--font-mono)', marginTop: '3px' }}>
                        O-D Travel Densities
                    </div>
                </div>

                {/* Arrow */}
                <div style={{ color: 'var(--accent-green)', display: 'flex', justifyContent: 'center' }}>
                    <ArrowRight size={18} />
                </div>

                {/* Stage 3 */}
                <div style={{
                    background: 'rgba(168,85,247,0.06)',
                    border: '1px solid rgba(168,85,247,0.25)',
                    borderRadius: '6px',
                    padding: '0.75rem',
                    textAlign: 'center',
                }}>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '4px' }}>
                        <Cpu size={16} color="var(--accent-purple)" />
                    </div>
                    <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>MODULE 3 INGESTION</div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
                        ST-GNN Predictor
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--accent-purple)', fontFamily: 'var(--font-mono)', marginTop: '3px' }}>
                        T+15m Horizon Forecast
                    </div>
                </div>
            </div>

            {/* Inferred Edge Flows & Origin-Destination Matrix */}
            <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                    <Layers size={13} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        AGGREGATE INTER-CAMERA CORRIDOR FLOWS (GRAPH EDGE WEIGHTS)
                    </span>
                </div>

                <div style={{
                    background: 'var(--bg-inset)',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    overflow: 'hidden',
                }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-color)', background: 'rgba(255,255,255,0.02)', color: 'var(--text-muted)' }}>
                                <th style={{ padding: '6px 10px', textAlign: 'left' }}>CAMERA CORRIDOR (EDGE)</th>
                                <th style={{ padding: '6px 10px', textAlign: 'center' }}>OBSERVED REID HOPS</th>
                                <th style={{ padding: '6px 10px', textAlign: 'center' }}>AVG TRANSIT TIME</th>
                                <th style={{ padding: '6px 10px', textAlign: 'right' }}>ESTIMATED FLOW RATE</th>
                            </tr>
                        </thead>
                        <tbody>
                            {odFlows.map((flow, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                                    <td style={{ padding: '6px 10px' }}>
                                        <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{flow.origin}</span>
                                        <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>→</span>
                                        <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{flow.destination}</span>
                                    </td>
                                    <td style={{ padding: '6px 10px', textAlign: 'center', color: 'var(--text-primary)', fontWeight: 700 }}>
                                        {flow.count} journeys
                                    </td>
                                    <td style={{ padding: '6px 10px', textAlign: 'center', color: 'var(--accent-amber)' }}>
                                        {flow.avgTravelTime}
                                    </td>
                                    <td style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--accent-green)', fontWeight: 700 }}>
                                        {flow.flowRate}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};
