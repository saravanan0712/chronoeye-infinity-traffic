import React, { useEffect, useRef, useState } from 'react';
import { Car, Gauge, AlertTriangle, Siren, Flame, TrendingUp, TrendingDown } from 'lucide-react';
import { NetworkSnapshot } from '../types/dashboard';

interface KpiPanelProps {
    snapshot: NetworkSnapshot;
    activeIncidentsCount: number;
    activeCorridorsCount: number;
}

// Animated counter hook
function useAnimatedValue(target: number, duration = 800) {
    const [displayed, setDisplayed] = useState(target);
    const prevRef = useRef(target);

    useEffect(() => {
        const start = prevRef.current;
        const end = target;
        if (start === end) return;
        const startTime = performance.now();

        const step = (now: number) => {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setDisplayed(start + (end - start) * eased);
            if (progress < 1) requestAnimationFrame(step);
            else prevRef.current = end;
        };
        requestAnimationFrame(step);
    }, [target, duration]);

    return displayed;
}

function KpiCard({
    title,
    rawValue,
    displayValue,
    unit,
    icon: Icon,
    color,
    trend,
    trendLabel,
    bgGlow,
    delay = 0,
}: {
    title: string;
    rawValue: number;
    displayValue: string;
    unit: string;
    icon: React.ElementType;
    color: string;
    trend?: 'up' | 'down' | 'neutral';
    trendLabel?: string;
    bgGlow?: string;
    delay?: number;
}) {
    return (
        <div
            className="glass-panel anim-slide-up"
            style={{
                padding: '1.1rem 1.25rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.6rem',
                animationDelay: `${delay}ms`,
                background: bgGlow
                    ? `linear-gradient(135deg, ${bgGlow} 0%, var(--bg-card) 100%)`
                    : 'var(--bg-card)',
                borderColor: color === 'var(--accent-rose)' ? 'rgba(255,24,96,0.25)'
                    : color === 'var(--accent-green)' ? 'rgba(0,255,157,0.2)'
                    : color === 'var(--accent-amber)' ? 'rgba(255,184,0,0.2)'
                    : 'var(--border-color)',
            }}
        >
            {/* Top Row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <span style={{
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    color: 'var(--text-secondary)',
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                    lineHeight: 1.2,
                }}>
                    {title}
                </span>
                <div style={{
                    padding: '5px',
                    borderRadius: '8px',
                    background: `${color}18`,
                    border: `1px solid ${color}30`,
                }}>
                    <Icon size={16} color={color} />
                </div>
            </div>

            {/* Value */}
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.35rem' }}>
                <span
                    className="kpi-value"
                    style={{
                        fontSize: '1.75rem',
                        fontWeight: 800,
                        color,
                        fontFamily: 'var(--font-mono)',
                        lineHeight: 1,
                        textShadow: `0 0 20px ${color}60`,
                    }}
                >
                    {displayValue}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                    {unit}
                </span>
            </div>

            {/* Progress Bar */}
            <div className="progress-bar-track">
                <div
                    className="progress-bar-fill"
                    style={{
                        width: `${Math.min(Math.abs(rawValue) * 2, 100)}%`,
                        background: `linear-gradient(90deg, ${color}80, ${color})`,
                    }}
                />
            </div>

            {/* Trend */}
            {trendLabel && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                    {trend === 'up' && <TrendingUp size={10} color="var(--accent-green)" />}
                    {trend === 'down' && <TrendingDown size={10} color="var(--accent-rose)" />}
                    <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>{trendLabel}</span>
                </div>
            )}
        </div>
    );
}

export const KpiPanel: React.FC<KpiPanelProps> = ({
    snapshot,
    activeIncidentsCount,
    activeCorridorsCount,
}) => {
    const vehicles = useAnimatedValue(snapshot.total_network_vehicles || 42);
    const speed = useAnimatedValue(snapshot.average_network_speed || 48.5);
    const congestion = useAnimatedValue((snapshot.network_congestion_index || 0.22) * 100);

    const congestionIndex = snapshot.network_congestion_index || 0.22;

    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))',
            gap: '0.85rem',
            marginBottom: '1rem',
        }}>
            <KpiCard
                title="Active Vehicles"
                rawValue={vehicles}
                displayValue={Math.round(vehicles).toString()}
                unit="veh"
                icon={Car}
                color="var(--accent-cyan)"
                trend="up"
                trendLabel="↑ +3 in last cycle"
                bgGlow="rgba(0,212,255,0.03)"
                delay={0}
            />
            <KpiCard
                title="Average Speed"
                rawValue={speed}
                displayValue={speed.toFixed(1)}
                unit="km/h"
                icon={Gauge}
                color="var(--accent-green)"
                trend="neutral"
                trendLabel="Within normal range"
                bgGlow="rgba(0,255,157,0.03)"
                delay={80}
            />
            <KpiCard
                title="Congestion Index"
                rawValue={congestion}
                displayValue={`${Math.round(congestion)}%`}
                unit="NCI"
                icon={Flame}
                color={congestionIndex > 0.6 ? 'var(--accent-rose)' : congestionIndex > 0.35 ? 'var(--accent-amber)' : 'var(--accent-green)'}
                trend={congestionIndex > 0.35 ? 'up' : 'down'}
                trendLabel={congestionIndex > 0.6 ? 'CRITICAL — Signal override recommended' : congestionIndex > 0.35 ? 'Moderate — monitoring' : 'Optimal flow'}
                bgGlow={congestionIndex > 0.6 ? 'rgba(255,24,96,0.04)' : 'rgba(255,184,0,0.02)'}
                delay={160}
            />
            <KpiCard
                title="Active Incidents"
                rawValue={activeIncidentsCount}
                displayValue={activeIncidentsCount.toString()}
                unit="alerts"
                icon={AlertTriangle}
                color={activeIncidentsCount > 0 ? 'var(--accent-rose)' : 'var(--text-muted)'}
                trend={activeIncidentsCount > 0 ? 'up' : 'neutral'}
                trendLabel={activeIncidentsCount > 0 ? 'Immediate attention required' : 'All clear'}
                bgGlow={activeIncidentsCount > 0 ? 'rgba(255,24,96,0.04)' : undefined}
                delay={240}
            />
            <KpiCard
                title="Green Corridors"
                rawValue={activeCorridorsCount}
                displayValue={activeCorridorsCount.toString()}
                unit="waves"
                icon={Siren}
                color={activeCorridorsCount > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)'}
                trendLabel={activeCorridorsCount > 0 ? 'Emergency preemption active' : 'Standby mode'}
                bgGlow={activeCorridorsCount > 0 ? 'rgba(0,212,255,0.05)' : undefined}
                delay={320}
            />
        </div>
    );
};
