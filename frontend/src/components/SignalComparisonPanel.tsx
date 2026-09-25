import React, { useState } from 'react';
import { Sliders, CheckCircle2, TrendingDown, BarChart2 } from 'lucide-react';

const CONTROLLERS = [
    {
        name: 'FIXED TIMING',
        type: 'BASELINE — Webster 1958',
        delay: 42.5,
        queue: 18,
        throughput: 380,
        travelTime: 65.0,
        stops: 8.2,
        color: '#4a5568',
        isBest: false,
        fillPercent: 42,
    },
    {
        name: 'REACTIVE ACTUATED',
        type: 'SENSOR-BASED ADAPTIVE',
        delay: 28.0,
        queue: 11,
        throughput: 440,
        travelTime: 48.0,
        stops: 5.1,
        color: '#ffb800',
        isBest: false,
        fillPercent: 65,
    },
    {
        name: 'CHRONOEYE PREDICTIVE',
        type: 'ST-GNN ROLLING HORIZON',
        delay: 15.8,
        queue: 4,
        throughput: 530,
        travelTime: 32.5,
        stops: 1.8,
        color: '#00d4ff',
        isBest: true,
        fillPercent: 100,
        improvement: '−62.8% DELAY REDUCTION',
    },
];

const METRICS = ['delay', 'queue', 'throughput', 'travelTime', 'stops'] as const;
type MetricKey = typeof METRICS[number];

export const SignalComparisonPanel: React.FC = () => {
    const [hoveredCtrl, setHoveredCtrl] = useState<number | null>(null);

    // Max values for bar scaling
    const maxVals: Record<MetricKey, number> = {
        delay: Math.max(...CONTROLLERS.map(c => c.delay)),
        queue: Math.max(...CONTROLLERS.map(c => c.queue)),
        throughput: Math.max(...CONTROLLERS.map(c => c.throughput)),
        travelTime: Math.max(...CONTROLLERS.map(c => c.travelTime)),
        stops: Math.max(...CONTROLLERS.map(c => c.stops)),
    };

    const getBarWidth = (key: MetricKey, val: number) => {
        return (val / maxVals[key]) * 100;
    };



    return (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Sliders size={18} color="var(--accent-cyan)" />
                    <div>
                        <h2 style={{
                            fontFamily: 'var(--font-display)',
                            fontSize: '0.75rem',
                            fontWeight: 700,
                            color: 'var(--text-primary)',
                            letterSpacing: '0.1em',
                        }}>
                            TRAFFIC SIGNAL CONTROLLER BENCHMARK
                        </h2>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            ChronoEye Predictive ST-GNN vs. Classical Baselines
                        </p>
                    </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <TrendingDown size={14} color="var(--accent-green)" />
                    <span style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                        −62.8% Total Network Delay
                    </span>
                </div>
            </div>

            {/* Controller Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
                {CONTROLLERS.map((ctrl, i) => (
                    <div
                        key={i}
                        onMouseEnter={() => setHoveredCtrl(i)}
                        onMouseLeave={() => setHoveredCtrl(null)}
                        style={{
                            background: ctrl.isBest
                                ? 'linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(0,212,255,0.05) 100%)'
                                : hoveredCtrl === i ? 'rgba(10,15,26,0.9)' : 'rgba(4,7,14,0.75)',
                            border: `1px solid ${ctrl.isBest ? 'rgba(0,212,255,0.5)' : hoveredCtrl === i ? 'rgba(0,212,255,0.2)' : 'var(--border-color)'}`,
                            borderRadius: '10px',
                            padding: '1.1rem',
                            position: 'relative',
                            transition: 'all 0.2s ease',
                            boxShadow: ctrl.isBest ? '0 0 24px rgba(0,212,255,0.1)' : 'none',
                        }}
                    >
                        {ctrl.isBest && (
                            <div style={{
                                position: 'absolute',
                                top: '-11px',
                                right: '12px',
                                background: 'var(--accent-cyan)',
                                color: '#000',
                                fontSize: '0.6rem',
                                fontWeight: 800,
                                padding: '3px 10px',
                                borderRadius: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                                letterSpacing: '0.04em',
                            }}>
                                <CheckCircle2 size={10} /> OPTIMAL
                            </div>
                        )}

                        <div style={{ marginBottom: '0.85rem' }}>
                            <h3 style={{
                                fontSize: '0.78rem',
                                fontWeight: 800,
                                color: ctrl.color,
                                fontFamily: 'var(--font-display)',
                                letterSpacing: '0.06em',
                                textShadow: ctrl.isBest ? `0 0 12px ${ctrl.color}60` : 'none',
                            }}>
                                {ctrl.name}
                            </h3>
                            <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>{ctrl.type}</span>
                        </div>

                        {/* Metrics */}
                        {[
                            { key: 'delay' as MetricKey, val: ctrl.delay, label: 'Delay', unit: 's/veh' },
                            { key: 'queue' as MetricKey, val: ctrl.queue, label: 'Queue', unit: 'veh' },
                            { key: 'throughput' as MetricKey, val: ctrl.throughput, label: 'Throughput', unit: 'v/h' },
                            { key: 'stops' as MetricKey, val: ctrl.stops, label: 'Stops', unit: '/veh' },
                        ].map(m => (
                            <div key={m.key} style={{ marginBottom: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                                    <span style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{m.label}</span>
                                    <span style={{
                                        fontSize: '0.75rem',
                                        fontFamily: 'var(--font-mono)',
                                        fontWeight: 700,
                                        color: ctrl.isBest ? ctrl.color : 'var(--text-primary)',
                                    }}>
                                        {m.val} <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem', fontWeight: 400 }}>{m.unit}</span>
                                    </span>
                                </div>
                                <div className="progress-bar-track">
                                    <div
                                        className="progress-bar-fill chart-bar"
                                        style={{
                                            width: `${getBarWidth(m.key, m.val)}%`,
                                            background: `linear-gradient(90deg, ${ctrl.color}50, ${ctrl.color})`,
                                        }}
                                    />
                                </div>
                            </div>
                        ))}

                        {ctrl.improvement && (
                            <div style={{
                                marginTop: '0.75rem',
                                paddingTop: '0.65rem',
                                borderTop: '1px solid rgba(0,212,255,0.2)',
                                fontSize: '0.68rem',
                                fontWeight: 800,
                                color: 'var(--accent-green)',
                                letterSpacing: '0.04em',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                            }}>
                                <TrendingDown size={12} />
                                {ctrl.improvement}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Comparison Bar Chart */}
            <div style={{
                background: 'var(--bg-inset)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                padding: '0.85rem',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.75rem' }}>
                    <BarChart2 size={12} color="var(--text-muted)" />
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                        Average Delay Comparison (s/vehicle)
                    </span>
                </div>
                {CONTROLLERS.map((ctrl, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.45rem' }}>
                        <span style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: ctrl.color, width: '160px', fontWeight: 700 }}>
                            {ctrl.name}
                        </span>
                        <div style={{ flex: 1, height: '16px', background: 'rgba(0,0,0,0.3)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div
                                className="chart-bar"
                                style={{
                                    width: `${(ctrl.delay / maxVals.delay) * 100}%`,
                                    height: '100%',
                                    background: `linear-gradient(90deg, ${ctrl.color}60, ${ctrl.color})`,
                                    borderRadius: '4px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    paddingLeft: '8px',
                                    fontSize: '0.65rem',
                                    color: '#000',
                                    fontWeight: 700,
                                    fontFamily: 'var(--font-mono)',
                                    transition: 'width 1s ease',
                                }}
                            >
                                {ctrl.delay}s
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
