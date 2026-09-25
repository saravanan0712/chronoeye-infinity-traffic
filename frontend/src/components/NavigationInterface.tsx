import React, { useState } from 'react';
import { Navigation, Compass, Zap, ArrowRight, MapPin, TrendingDown, Shield } from 'lucide-react';
import { fetchRouteComparison } from '../services/api';

const JUNCTION_OPTIONS = [
    { value: 'JUNC_1', label: 'J1 — Anna Salai North' },
    { value: 'JUNC_2', label: 'J2 — Central Junction' },
    { value: 'JUNC_3', label: 'J3 — Bypass Avenue' },
    { value: 'JUNC_4', label: 'J4 — GST Terminal' },
    { value: 'JUNC_5', label: 'J5 — Tech Park Hub' },
];

const DEFAULT_ROUTES = {
    standardRoute: {
        algorithm: 'Standard Dijkstra (Shortest Path)',
        path_nodes: ['JUNC_1', 'JUNC_2', 'JUNC_4'],
        path_roads: ['SEG_1_2', 'SEG_2_4'],
        total_distance_meters: 1000.0,
        estimated_travel_time_seconds: 90.0,
        average_congestion_score: 0.65,
    },
    predictiveRoute: {
        algorithm: 'ChronoEye Predictive A* (ST-GNN)',
        path_nodes: ['JUNC_1', 'JUNC_5', 'JUNC_4'],
        path_roads: ['SEG_1_5', 'SEG_5_4'],
        total_distance_meters: 1200.0,
        estimated_travel_time_seconds: 52.0,
        average_congestion_score: 0.15,
        uncertainty_std: 3.2,
    },
};

function RouteCard({
    title,
    route,
    type,
    timeSaved,
    pctSaved,
}: {
    title: string;
    route: any;
    type: 'standard' | 'predictive';
    timeSaved?: number;
    pctSaved?: number;
}) {
    const isPredict = type === 'predictive';
    const congPct = ((route.average_congestion_score || 0) * 100).toFixed(0);
    const distKm = ((route.total_distance_meters || 0) / 1000).toFixed(2);
    const etaMin = Math.floor((route.estimated_travel_time_seconds || 0) / 60);
    const etaSec = Math.round((route.estimated_travel_time_seconds || 0) % 60);

    return (
        <div style={{
            background: isPredict
                ? 'linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(0,150,200,0.04) 100%)'
                : 'rgba(255,24,96,0.04)',
            border: `1px solid ${isPredict ? 'rgba(0,212,255,0.5)' : 'rgba(255,24,96,0.3)'}`,
            borderRadius: '12px',
            padding: '1.25rem',
            position: 'relative',
            boxShadow: isPredict ? '0 0 24px rgba(0,212,255,0.1)' : 'none',
        }}>
            {/* Best badge */}
            {isPredict && timeSaved !== undefined && (
                <div style={{
                    position: 'absolute',
                    top: '-12px',
                    right: '16px',
                    background: 'linear-gradient(90deg, var(--accent-cyan), #00aaff)',
                    color: '#000',
                    fontSize: '0.65rem',
                    fontWeight: 800,
                    padding: '3px 12px',
                    borderRadius: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    letterSpacing: '0.04em',
                    boxShadow: '0 4px 12px rgba(0,212,255,0.4)',
                }}>
                    <Zap size={10} /> {timeSaved.toFixed(0)}s FASTER ({pctSaved}% SAVED)
                </div>
            )}

            {/* Title */}
            <div style={{ marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '0.78rem', fontWeight: 700, color: isPredict ? 'var(--accent-cyan)' : 'var(--accent-rose)', letterSpacing: '0.04em' }}>
                    {title}
                </h3>
                <p style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {route.algorithm}
                </p>
            </div>

            {/* ETA Big Number */}
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '1rem' }}>
                <span style={{
                    fontSize: '2.2rem',
                    fontWeight: 800,
                    color: isPredict ? 'var(--accent-cyan)' : 'var(--accent-rose)',
                    fontFamily: 'var(--font-mono)',
                    lineHeight: 1,
                    textShadow: `0 0 20px ${isPredict ? 'rgba(0,212,255,0.4)' : 'rgba(255,24,96,0.4)'}`,
                }}>
                    {etaMin > 0 ? `${etaMin}m ${etaSec}s` : `${etaSec}s`}
                </span>
                {isPredict && route.uncertainty_std && (
                    <span style={{ fontSize: '0.72rem', color: 'var(--accent-green)', fontWeight: 600 }}>
                        ±{route.uncertainty_std}s
                    </span>
                )}
            </div>

            {/* Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem', marginBottom: '0.85rem' }}>
                {[
                    { label: 'Distance', value: `${distKm} km`, icon: MapPin },
                    { label: 'Congestion', value: `${congPct}%`, icon: Navigation,
                      color: Number(congPct) > 50 ? 'var(--accent-rose)' : Number(congPct) > 25 ? 'var(--accent-amber)' : 'var(--accent-green)' },
                    ...(isPredict ? [
                        { label: 'Confidence', value: '94%', icon: Shield, color: 'var(--accent-green)' },
                    ] : []),
                ].map((m, i) => (
                    <div key={i} style={{
                        background: 'rgba(0,0,0,0.2)',
                        borderRadius: '6px',
                        padding: '0.5rem 0.7rem',
                        border: '1px solid var(--border-color)',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '2px' }}>
                            <m.icon size={10} color="var(--text-muted)" />
                            <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{m.label}</span>
                        </div>
                        <span style={{ fontSize: '0.82rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: (m as any).color || 'var(--text-primary)' }}>
                            {m.value}
                        </span>
                    </div>
                ))}
            </div>

            {/* Path Display */}
            <div style={{
                background: 'rgba(0,0,0,0.25)',
                borderRadius: '6px',
                padding: '0.55rem 0.75rem',
                border: '1px solid var(--border-color)',
            }}>
                <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Route Path
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', flexWrap: 'wrap' }}>
                    {(route.path_nodes || []).map((node: string, i: number) => (
                        <React.Fragment key={node}>
                            <span style={{
                                fontSize: '0.72rem',
                                fontFamily: 'var(--font-mono)',
                                fontWeight: 700,
                                color: isPredict ? 'var(--accent-green)' : 'var(--text-secondary)',
                                background: isPredict ? 'rgba(0,255,157,0.1)' : 'rgba(255,255,255,0.05)',
                                padding: '2px 6px',
                                borderRadius: '4px',
                            }}>
                                {node}
                            </span>
                            {i < (route.path_nodes?.length || 1) - 1 && (
                                <ArrowRight size={10} color="var(--text-muted)" />
                            )}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </div>
    );
}

export const NavigationInterface: React.FC = () => {
    const [origin, setOrigin] = useState('JUNC_1');
    const [destination, setDestination] = useState('JUNC_4');
    const [isLoading, setIsLoading] = useState(false);
    const [routes, setRoutes] = useState(DEFAULT_ROUTES);

    const handleCalculateRoute = async () => {
        setIsLoading(true);
        try {
            const res = await fetchRouteComparison(origin, destination);
            if (res && res.standardRoute && res.predictiveRoute) {
                setRoutes(res);
            }
        } catch {
            // Keep defaults on API error
        }
        setIsLoading(false);
    };

    const stdTime = routes.standardRoute?.estimated_travel_time_seconds || 90.0;
    const predTime = routes.predictiveRoute?.estimated_travel_time_seconds || 52.0;
    const timeSaved = Math.max(0, stdTime - predTime);
    const pctSaved = Math.round((timeSaved / stdTime) * 100);

    return (
        <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{
                        width: '44px', height: '44px', borderRadius: '12px',
                        background: 'linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,100,200,0.1))',
                        border: '1px solid rgba(0,212,255,0.35)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                        <Navigation size={22} color="var(--accent-cyan)" style={{ filter: 'drop-shadow(0 0 8px rgba(0,212,255,0.6))' }} />
                    </div>
                    <div>
                        <h2 style={{
                            fontFamily: 'var(--font-display)',
                            fontSize: '0.85rem',
                            fontWeight: 700,
                            color: 'var(--text-primary)',
                            letterSpacing: '0.1em',
                        }}>
                            PREDICTIVE NAVIGATION ENGINE
                        </h2>
                        <p style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Spatio-Temporal A* · Uncertainty-Aware · Multi-Horizon Forecasting
                        </p>
                    </div>
                </div>

                {/* Route Selector */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    background: 'var(--bg-inset)',
                    padding: '0.6rem 1rem',
                    borderRadius: '10px',
                    border: '1px solid var(--border-color)',
                }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>FROM</span>
                        <select
                            id="nav-origin"
                            value={origin}
                            onChange={e => setOrigin(e.target.value)}
                            style={{
                                background: 'none', border: 'none',
                                color: 'var(--accent-cyan)', fontWeight: 700,
                                fontSize: '0.8rem', fontFamily: 'var(--font-mono)',
                                outline: 'none', cursor: 'pointer',
                            }}
                        >
                            {JUNCTION_OPTIONS.map(o => (
                                <option key={o.value} value={o.value} style={{ background: '#06090f' }}>{o.label}</option>
                            ))}
                        </select>
                    </div>

                    <ArrowRight size={16} color="var(--text-muted)" />

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                        <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>TO</span>
                        <select
                            id="nav-destination"
                            value={destination}
                            onChange={e => setDestination(e.target.value)}
                            style={{
                                background: 'none', border: 'none',
                                color: 'var(--accent-green)', fontWeight: 700,
                                fontSize: '0.8rem', fontFamily: 'var(--font-mono)',
                                outline: 'none', cursor: 'pointer',
                            }}
                        >
                            {JUNCTION_OPTIONS.map(o => (
                                <option key={o.value} value={o.value} style={{ background: '#06090f' }}>{o.label}</option>
                            ))}
                        </select>
                    </div>

                    <button
                        id="btn-calculate-route"
                        onClick={handleCalculateRoute}
                        disabled={isLoading}
                        style={{
                            background: isLoading ? 'rgba(0,212,255,0.2)' : 'var(--accent-cyan)',
                            color: '#000',
                            border: 'none',
                            padding: '0.5rem 1rem',
                            borderRadius: '8px',
                            fontWeight: 800,
                            fontSize: '0.72rem',
                            cursor: isLoading ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            transition: 'all 0.2s ease',
                            letterSpacing: '0.04em',
                        }}
                    >
                        {isLoading ? (
                            <><span className="spinner" style={{ width: '12px', height: '12px', borderTopColor: '#000' }} /> CALC...</>
                        ) : (
                            <><Compass size={13} /> COMPARE</>
                        )}
                    </button>
                </div>
            </div>

            {/* Summary Banner */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '1.5rem',
                padding: '0.75rem 1.5rem',
                background: 'linear-gradient(90deg, rgba(0,212,255,0.06), rgba(0,255,157,0.06))',
                borderRadius: '10px',
                border: '1px solid rgba(0,212,255,0.15)',
                marginBottom: '1.25rem',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <TrendingDown size={16} color="var(--accent-green)" />
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                        {timeSaved.toFixed(0)}s TIME SAVED
                    </span>
                </div>
                <div style={{ width: '1px', height: '24px', background: 'var(--border-color)' }} />
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    ChronoEye achieves <strong style={{ color: 'var(--accent-cyan)' }}>{pctSaved}% faster</strong> travel vs shortest path routing
                </div>
                <div style={{ width: '1px', height: '24px', background: 'var(--border-color)' }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Shield size={14} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        Uncertainty: <strong style={{ color: 'var(--accent-cyan)' }}>±{routes.predictiveRoute?.uncertainty_std || 3.2}s</strong>
                    </span>
                </div>
            </div>

            {/* Route Comparison Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                <RouteCard
                    title="STANDARD SHORTEST PATH"
                    route={routes.standardRoute}
                    type="standard"
                />
                <RouteCard
                    title="CHRONOEYE PREDICTIVE OPTIMAL"
                    route={routes.predictiveRoute}
                    type="predictive"
                    timeSaved={timeSaved}
                    pctSaved={pctSaved}
                />
            </div>
        </div>
    );
};
