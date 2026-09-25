import React, { useState, useEffect } from 'react';
import { ShieldAlert, Radio, Play, Pause, RotateCcw, Clock } from 'lucide-react';
import { SystemStatus } from '../types/dashboard';

interface HeaderProps {
    systemStatus: SystemStatus;
    isDemoMode: boolean;
    isSimulating: boolean;
    onToggleSim: () => void;
    onResetSim: () => void;
}

export const Header: React.FC<HeaderProps> = ({
    systemStatus, isDemoMode, isSimulating, onToggleSim, onResetSim,
}) => {
    const [time, setTime] = useState(new Date());
    const [frameCount, setFrameCount] = useState(0);

    useEffect(() => {
        const t = setInterval(() => {
            setTime(new Date());
            setFrameCount(p => p + 1);
        }, 1000);
        return () => clearInterval(t);
    }, []);

    const moduleColor = (s: string) => {
        if (s === 'ONLINE' || s === 'LIVE') return 'var(--accent-green)';
        if (s === 'DEGRADED' || s === 'DEMO' || s === 'DEMO DATA') return 'var(--accent-amber)';
        return 'var(--accent-rose)'; // OFFLINE, NOT CONNECTED, NOT_CONNECTED
    };

    return (
        <div className="glass-panel" style={{ padding: '0.75rem 1.25rem', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                {/* Logo */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <ShieldAlert size={24} color="var(--accent-cyan)" style={{ filter: 'drop-shadow(0 0 8px rgba(0,212,255,0.6))' }} />
                    <div>
                        <h1 style={{
                            fontFamily: 'var(--font-display)', fontSize: '0.95rem', fontWeight: 900,
                            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-purple))',
                            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                            letterSpacing: '0.12em',
                        }}>
                            CHRONOEYE INFINITY
                        </h1>
                        <p style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.08em', marginTop: '1px' }}>
                            MULTI-SIGNAL VEHICLE INTELLIGENCE
                        </p>
                    </div>
                </div>

                {/* Module Status */}
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                    {[
                        { label: 'MODULE 1', sub: 'PERCEPTION', status: systemStatus.module1Status },
                        { label: 'MODULE 2', sub: 'RE-ID', status: systemStatus.module2Status },
                        { label: 'MODULE 3', sub: 'GRAPH', status: systemStatus.module3Status },
                    ].map(m => (
                        <div key={m.label} style={{
                            display: 'flex', alignItems: 'center', gap: '0.35rem',
                            padding: '0.3rem 0.6rem', borderRadius: '6px',
                            background: 'var(--bg-inset)', border: '1px solid var(--border-color)',
                        }}>
                            <div style={{
                                width: '6px', height: '6px', borderRadius: '50%',
                                background: moduleColor(m.status),
                                boxShadow: `0 0 6px ${moduleColor(m.status)}80`,
                            }} />
                            <div>
                                <div style={{ fontSize: '0.55rem', fontWeight: 700, color: moduleColor(m.status), letterSpacing: '0.04em' }}>
                                    {m.label}
                                </div>
                                <div style={{ fontSize: '0.48rem', color: 'var(--text-muted)' }}>{m.sub}</div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Live/Demo + Cameras + Time */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    {/* Data Source Badge */}
                    <div className={`demo-mode-badge ${isDemoMode ? 'demo' : 'live'}`}>
                        <div style={{
                            width: '6px', height: '6px', borderRadius: '50%',
                            background: isDemoMode ? 'var(--accent-amber)' : 'var(--accent-green)',
                        }} />
                        {isDemoMode ? 'DEMO MODE' : 'LIVE DATA'}
                    </div>

                    {/* Camera Count */}
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '0.3rem',
                        fontSize: '0.65rem', color: 'var(--text-secondary)',
                    }}>
                        <Radio size={11} />
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                            {systemStatus.activeCameraCount}
                        </span>
                        <span>CAMERAS</span>
                    </div>

                    {/* Sim Controls */}
                    <div style={{ display: 'flex', gap: '4px' }}>
                        <button onClick={onToggleSim} style={{
                            background: 'var(--bg-inset)', border: '1px solid var(--border-color)',
                            color: 'var(--text-secondary)', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: '3px', fontSize: '0.58rem',
                        }}>
                            {isSimulating ? <Pause size={10} /> : <Play size={10} />}
                            {isSimulating ? 'PAUSE' : 'PLAY'}
                        </button>
                        <button onClick={onResetSim} style={{
                            background: 'var(--bg-inset)', border: '1px solid var(--border-color)',
                            color: 'var(--text-secondary)', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer',
                            display: 'flex', alignItems: 'center', fontSize: '0.58rem',
                        }}>
                            <RotateCcw size={10} />
                        </button>
                    </div>

                    {/* Clock */}
                    <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.82rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
                            <Clock size={11} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                            {time.toLocaleTimeString()}
                        </div>
                        <div style={{ fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                            {time.toLocaleDateString()} · F{frameCount.toString().padStart(5, '0')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
