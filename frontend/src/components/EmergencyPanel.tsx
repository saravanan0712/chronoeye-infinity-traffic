import React, { useState } from 'react';
import { Siren, Radio, Zap, ArrowRight } from 'lucide-react';
import { EmergencyCorridorPlan } from '../types/dashboard';

interface EmergencyPanelProps {
    corridor: EmergencyCorridorPlan | null;
    onTriggerCorridor: (vehId: string, origin: string, dest: string) => void;
}

const VEHICLE_PRESETS = [
    { id: 'AMB_911',  label: '🚑 Ambulance', color: '#ff1860' },
    { id: 'FIRE_101', label: '🚒 Fire Engine', color: '#ff8800' },
    { id: 'POL_42',   label: '🚓 Police', color: '#0099ff' },
    { id: 'VVIP_001', label: '🚖 VVIP Escort', color: '#cc00ff' },
];

const JUNCTION_OPTIONS = [
    { value: 'JUNC_1', label: 'J1 — Anna Salai North' },
    { value: 'JUNC_2', label: 'J2 — Central Junction' },
    { value: 'JUNC_3', label: 'J3 — Bypass Avenue' },
    { value: 'JUNC_4', label: 'J4 — GST Terminal' },
    { value: 'JUNC_5', label: 'J5 — Tech Park Hub' },
];

export const EmergencyPanel: React.FC<EmergencyPanelProps> = ({ corridor, onTriggerCorridor }) => {
    const [vehicleId, setVehicleId] = useState('AMB_911');
    const [origin, setOrigin] = useState('JUNC_1');
    const [dest, setDest] = useState('JUNC_4');
    const [isDispatching, setIsDispatching] = useState(false);
    const [selectedPreset, setSelectedPreset] = useState(0);

    const handleDispatch = async () => {
        setIsDispatching(true);
        await onTriggerCorridor(vehicleId, origin, dest);
        setTimeout(() => setIsDispatching(false), 2000);
    };

    return (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Siren size={18} color="var(--accent-rose)" style={{ filter: 'drop-shadow(0 0 6px rgba(255,24,96,0.6))' }} />
                <div>
                    <h2 style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        color: 'var(--text-primary)',
                        letterSpacing: '0.1em',
                    }}>
                        EMERGENCY GREEN CORRIDOR PREEMPTION
                    </h2>
                    <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                        Signal preemption · Priority routing · Optimal pathway calculation
                    </p>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                {/* Dispatch Form */}
                <div style={{
                    background: 'var(--bg-inset)',
                    border: '1px solid rgba(0,212,255,0.12)',
                    borderRadius: '10px',
                    padding: '1rem',
                }}>
                    <h3 style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '0.85rem', letterSpacing: '0.06em' }}>
                        DISPATCH CONFIGURATION
                    </h3>

                    {/* Vehicle Type Presets */}
                    <div style={{ marginBottom: '0.85rem' }}>
                        <label style={{ display: 'block', fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: '0.4rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                            Vehicle Type
                        </label>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem' }}>
                            {VEHICLE_PRESETS.map((veh, i) => (
                                <button
                                    key={veh.id}
                                    onClick={() => { setSelectedPreset(i); setVehicleId(veh.id); }}
                                    style={{
                                        background: selectedPreset === i ? `${veh.color}20` : 'transparent',
                                        border: `1px solid ${selectedPreset === i ? veh.color : 'var(--border-color)'}`,
                                        color: selectedPreset === i ? veh.color : 'var(--text-secondary)',
                                        padding: '0.35rem 0.5rem',
                                        borderRadius: '6px',
                                        fontSize: '0.65rem',
                                        fontWeight: 600,
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease',
                                        textAlign: 'left',
                                    }}
                                >
                                    {veh.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Vehicle ID Input */}
                    <div style={{ marginBottom: '0.75rem' }}>
                        <label style={{ display: 'block', fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                            Unit ID
                        </label>
                        <input
                            id="emergency-vehicle-id"
                            type="text"
                            value={vehicleId}
                            onChange={(e) => setVehicleId(e.target.value)}
                            style={{
                                width: '100%',
                                background: 'rgba(6,9,15,0.9)',
                                border: '1px solid var(--border-color)',
                                color: '#fff',
                                padding: '0.45rem 0.7rem',
                                borderRadius: '6px',
                                fontSize: '0.8rem',
                                fontFamily: 'var(--font-mono)',
                                transition: 'border-color 0.2s ease',
                            }}
                        />
                    </div>

                    {/* Origin → Destination */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '0.35rem', alignItems: 'center', marginBottom: '1rem' }}>
                        <div>
                            <label style={{ display: 'block', fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                Origin
                            </label>
                            <select
                                id="emergency-origin"
                                value={origin}
                                onChange={(e) => setOrigin(e.target.value)}
                                style={{
                                    width: '100%',
                                    background: 'rgba(6,9,15,0.9)',
                                    border: '1px solid var(--border-color)',
                                    color: 'var(--accent-cyan)',
                                    padding: '0.4rem 0.5rem',
                                    borderRadius: '6px',
                                    fontSize: '0.72rem',
                                    fontFamily: 'var(--font-mono)',
                                    fontWeight: 700,
                                }}
                            >
                                {JUNCTION_OPTIONS.map(o => (
                                    <option key={o.value} value={o.value} style={{ background: '#06090f' }}>{o.label}</option>
                                ))}
                            </select>
                        </div>
                        <ArrowRight size={14} color="var(--text-muted)" style={{ marginTop: '16px' }} />
                        <div>
                            <label style={{ display: 'block', fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: '4px', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                                Destination
                            </label>
                            <select
                                id="emergency-dest"
                                value={dest}
                                onChange={(e) => setDest(e.target.value)}
                                style={{
                                    width: '100%',
                                    background: 'rgba(6,9,15,0.9)',
                                    border: '1px solid var(--border-color)',
                                    color: 'var(--accent-green)',
                                    padding: '0.4rem 0.5rem',
                                    borderRadius: '6px',
                                    fontSize: '0.72rem',
                                    fontFamily: 'var(--font-mono)',
                                    fontWeight: 700,
                                }}
                            >
                                {JUNCTION_OPTIONS.map(o => (
                                    <option key={o.value} value={o.value} style={{ background: '#06090f' }}>{o.label}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Dispatch Button */}
                    <button
                        id="btn-dispatch-corridor"
                        onClick={handleDispatch}
                        disabled={isDispatching}
                        style={{
                            width: '100%',
                            background: isDispatching
                                ? 'rgba(255,24,96,0.3)'
                                : 'linear-gradient(135deg, #ff1860, #ff5500)',
                            color: '#fff',
                            border: 'none',
                            padding: '0.65rem',
                            borderRadius: '8px',
                            fontWeight: 800,
                            fontSize: '0.78rem',
                            cursor: isDispatching ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '0.5rem',
                            letterSpacing: '0.05em',
                            boxShadow: isDispatching ? 'none' : '0 4px 16px rgba(255,24,96,0.4)',
                            transition: 'all 0.2s ease',
                        }}
                    >
                        {isDispatching ? (
                            <>
                                <span className="spinner" style={{ width: '14px', height: '14px', borderTopColor: '#fff' }} />
                                CALCULATING PATH...
                            </>
                        ) : (
                            <>
                                <Zap size={14} />
                                DISPATCH GREEN WAVE
                            </>
                        )}
                    </button>
                </div>

                {/* Active Corridor Monitor */}
                <div style={{
                    background: corridor ? 'rgba(0,212,255,0.05)' : 'var(--bg-inset)',
                    border: `1px solid ${corridor ? 'rgba(0,212,255,0.4)' : 'var(--border-color)'}`,
                    borderRadius: '10px',
                    padding: '1rem',
                    boxShadow: corridor ? '0 0 20px rgba(0,212,255,0.08)' : 'none',
                    transition: 'all 0.5s ease',
                }}>
                    <h3 style={{ fontSize: '0.72rem', fontWeight: 700, color: corridor ? 'var(--accent-green)' : 'var(--text-muted)', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.35rem', letterSpacing: '0.06em' }}>
                        <Radio size={13} className={corridor ? 'status-online' : 'status-offline'} />
                        ACTIVE CORRIDOR MONITOR
                    </h3>

                    {corridor ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {[
                                { label: 'Corridor ID', value: corridor.corridor_id, color: 'var(--accent-cyan)' },
                                { label: 'Vehicle', value: corridor.vehicle_id, color: '#fff' },
                                { label: 'Route', value: corridor.ordered_junctions?.join(' → '), color: 'var(--accent-green)' },
                                { label: 'ETA', value: `${corridor.estimated_total_travel_time_seconds}s`, color: 'var(--accent-cyan)' },
                                { label: 'Status', value: corridor.status, color: 'var(--accent-green)' },
                            ].map(row => (
                                <div key={row.label}>
                                    <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: '2px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                        {row.label}
                                    </div>
                                    <div style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: row.color }}>
                                        {row.value}
                                    </div>
                                </div>
                            ))}

                            {/* Animated progress */}
                            <div style={{ marginTop: '0.5rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                    <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>SIGNAL PREEMPTION PROGRESS</span>
                                    <span style={{ fontSize: '0.62rem', color: 'var(--accent-cyan)' }}>IN TRANSIT</span>
                                </div>
                                <div className="progress-bar-track">
                                    <div
                                        className="progress-bar-fill"
                                        style={{
                                            width: '65%',
                                            background: 'linear-gradient(90deg, rgba(0,212,255,0.5), var(--accent-cyan))',
                                            boxShadow: '0 0 8px rgba(0,212,255,0.5)',
                                        }}
                                    />
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', minHeight: '140px' }}>
                            <Siren size={32} color="rgba(0,212,255,0.2)" />
                            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center', letterSpacing: '0.04em' }}>
                                NO EMERGENCY CORRIDOR ACTIVE
                            </span>
                            <span className="badge badge-cyan">STANDBY</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
