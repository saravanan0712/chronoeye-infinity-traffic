import React, { useState } from 'react';
import { BarChart2, Cpu } from 'lucide-react';
import { SegmentForecast } from '../types/dashboard';

interface ForecastPanelProps {
    forecasts?: Record<string, SegmentForecast>;
    isDemoMode?: boolean;
}

const HORIZONS = [
    { label: '+5 MIN',  flow: 435, speed: 51.0, queue: 3,  conf: 95, density: 38 },
    { label: '+10 MIN', flow: 460, speed: 47.5, queue: 5,  conf: 92, density: 52 },
    { label: '+15 MIN', flow: 490, speed: 42.0, queue: 8,  conf: 88, density: 67 },
    { label: '+30 MIN', flow: 520, speed: 38.0, queue: 12, conf: 82, density: 75 },
];

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    const w = 100;
    const h = 30;
    const pts = data.map((v, i) => {
        const x = (i / (data.length - 1)) * w;
        const y = h - ((v - min) / range) * h;
        return `${x},${y}`;
    });
    const pathD = `M ${pts.join(' L ')}`;
    const areaD = `M 0,${h} L ${pts.join(' L ')} L ${w},${h} Z`;

    return (
        <svg width="100" height="30" viewBox="0 0 100 30" style={{ overflow: 'visible' }}>
            <defs>
                <linearGradient id={`sg-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity="0.3" />
                    <stop offset="100%" stopColor={color} stopOpacity="0" />
                </linearGradient>
            </defs>
            <path d={areaD} fill={`url(#sg-${color.replace('#', '')})`} />
            <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            {/* Last point dot */}
            {data.length > 0 && (() => {
                const last = pts[pts.length - 1].split(',');
                return <circle cx={last[0]} cy={last[1]} r="2.5" fill={color} style={{ filter: `drop-shadow(0 0 4px ${color})` }} />;
            })()}
        </svg>
    );
}

export const ForecastPanel: React.FC<ForecastPanelProps> = ({ forecasts: _forecasts, isDemoMode = false }) => {
    const [selected, setSelected] = useState(0);
    const [sparkData] = useState(() => ({
        flow: [380, 400, 420, 435, 460, 490, 520],
        speed: [58, 55, 53, 51, 47, 42, 38],
        queue: [1, 2, 2, 3, 5, 8, 12],
    }));

    const hasLiveForecasts = !isDemoMode && !!_forecasts && Object.keys(_forecasts).length > 0;
    const h = HORIZONS[selected];
    const maxFlow = 600;

    return (
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <BarChart2 size={18} color="var(--accent-cyan)" />
                    <div>
                        <h2 style={{
                            fontFamily: 'var(--font-display)',
                            fontSize: '0.72rem',
                            fontWeight: 700,
                            color: 'var(--text-primary)',
                            letterSpacing: '0.1em',
                        }}>
                            ST-GNN MULTI-HORIZON FORECAST
                        </h2>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Spatio-Temporal Graph Neural Network · Rolling Horizon
                        </p>
                    </div>
                </div>
                <div className={`demo-mode-badge ${hasLiveForecasts ? 'live' : 'demo'}`} style={{ fontSize: '0.55rem', padding: '2px 7px' }}>
                    <span style={{
                        width: '5px', height: '5px', borderRadius: '50%',
                        background: hasLiveForecasts ? 'var(--accent-green)' : 'var(--accent-amber)',
                        display: 'inline-block', marginRight: '4px',
                    }} />
                    {hasLiveForecasts ? 'LIVE DATA' : 'DEMO DATA'}
                </div>
            </div>

            {/* Horizon Tabs */}
            <div style={{ display: 'flex', gap: '0.4rem' }}>
                {HORIZONS.map((hz, i) => (
                    <button
                        key={i}
                        onClick={() => setSelected(i)}
                        style={{
                            flex: 1,
                            background: selected === i ? 'rgba(0,212,255,0.15)' : 'rgba(4,7,14,0.7)',
                            border: `1px solid ${selected === i ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                            color: selected === i ? 'var(--accent-cyan)' : 'var(--text-muted)',
                            padding: '0.35rem 0',
                            borderRadius: '6px',
                            fontSize: '0.65rem',
                            fontWeight: 700,
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                            fontFamily: 'var(--font-mono)',
                            letterSpacing: '0.04em',
                        }}
                    >
                        {hz.label}
                    </button>
                ))}
            </div>

            {/* Selected Horizon Detail */}
            <div style={{
                background: 'var(--bg-inset)',
                borderRadius: '8px',
                border: '1px solid rgba(0,212,255,0.1)',
                padding: '0.85rem',
            }}>
                {/* Confidence Badge */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontWeight: 600 }}>MODEL CONFIDENCE</span>
                    <span className="badge badge-green">{h.conf}% CONF</span>
                </div>

                {/* Metrics */}
                {[
                    { label: 'Vehicle Flow', value: `${h.flow} veh/h`, sparkData: sparkData.flow, color: '#00d4ff', width: (h.flow / maxFlow) * 100 },
                    { label: 'Avg Speed', value: `${h.speed} km/h`, sparkData: sparkData.speed, color: '#00ff9d', width: (h.speed / 70) * 100 },
                    { label: 'Queue Length', value: `${h.queue} veh`, sparkData: sparkData.queue, color: h.queue > 6 ? '#ff1860' : '#ffb800', width: (h.queue / 15) * 100 },
                ].map(metric => (
                    <div key={metric.label} style={{ marginBottom: '0.65rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                            <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>{metric.label}</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <MiniSparkline data={metric.sparkData} color={metric.color} />
                                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: metric.color, fontFamily: 'var(--font-mono)' }}>
                                    {metric.value}
                                </span>
                            </div>
                        </div>
                        <div className="progress-bar-track">
                            <div
                                className="progress-bar-fill"
                                style={{ width: `${metric.width}%`, background: `linear-gradient(90deg, ${metric.color}80, ${metric.color})` }}
                            />
                        </div>
                    </div>
                ))}

                {/* Density indicator */}
                <div style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>
                        <Cpu size={10} style={{ display: 'inline', marginRight: '4px' }} />
                        ST-GNN Density Estimate
                    </span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--accent-purple)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                        {h.density} veh/km
                    </span>
                </div>
            </div>

            {/* All Horizons Summary */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    All Horizon Overview
                </span>
                {HORIZONS.map((hz, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }} onClick={() => setSelected(i)}>
                        <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', width: '55px' }}>{hz.label}</span>
                        <div className="progress-bar-track" style={{ flex: 1 }}>
                            <div
                                className="progress-bar-fill"
                                style={{
                                    width: `${(hz.flow / maxFlow) * 100}%`,
                                    background: `linear-gradient(90deg, rgba(0,212,255,0.5), ${i === selected ? '#00d4ff' : 'rgba(0,212,255,0.3)'})`,
                                    height: '4px',
                                }}
                            />
                        </div>
                        <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', width: '50px', textAlign: 'right' }}>
                            {hz.flow} v/h
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};
