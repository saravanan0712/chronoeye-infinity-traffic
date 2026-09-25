import React from 'react';
import { Camera } from 'lucide-react';
import { CameraMetadata, ActiveVehicleTrack } from '../types/dashboard';

interface CameraGridProps {
    cameras: CameraMetadata[];
    tracksByCamera: Record<string, ActiveVehicleTrack[]>;
    selectedTrackId?: string;
    isDemoMode: boolean;
    onSelectTrack?: (trackId: string, cameraId?: string) => void;
}

const statusClass = (s: string) => {
    if (s === 'LIVE') return 'online';
    if (s === 'SIMULATION') return 'simulation';
    return 'offline';
};

const statusLabel = (s: string) => {
    if (s === 'LIVE') return 'ONLINE';
    if (s === 'SIMULATION') return 'SIMULATION';
    if (s === 'PLAYBACK') return 'PLAYBACK';
    return 'OFFLINE';
};

export const CameraGrid: React.FC<CameraGridProps> = ({ cameras, tracksByCamera, isDemoMode, onSelectTrack }) => {
    return (
        <div className="glass-panel" style={{ padding: '1rem', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.85rem' }}>
                <Camera size={16} color="var(--accent-cyan)" />
                <h2 style={{
                    fontFamily: 'var(--font-display)', fontSize: '0.72rem', fontWeight: 700,
                    color: 'var(--text-primary)', letterSpacing: '0.1em',
                }}>
                    LIVE MULTI-CAMERA OBSERVATIONS
                </h2>
                {isDemoMode && <span className="demo-mode-badge demo" style={{ fontSize: '0.5rem', padding: '1px 6px' }}>DEMO DATA</span>}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(cameras.length, 4)}, 1fr)`, gap: '0.65rem' }}>
                {cameras.map(cam => {
                    const tracks = tracksByCamera[cam.camera_id] || [];
                    return (
                        <div key={cam.camera_id} style={{
                            background: 'var(--bg-inset)', borderRadius: '10px',
                            border: '1px solid var(--border-color)', padding: '0.85rem',
                            transition: 'border-color 0.2s',
                        }}>
                            {/* Camera Header */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                    <div className={`cam-status-dot ${statusClass(cam.status)}`} />
                                    <span style={{
                                        fontSize: '0.78rem', fontWeight: 800, fontFamily: 'var(--font-mono)',
                                        color: 'var(--accent-cyan)',
                                    }}>
                                        {cam.camera_id}
                                    </span>
                                </div>
                                <span className={`decision-badge decision-${cam.status === 'LIVE' ? 'MATCH_CONFIRMED' : cam.status === 'SIMULATION' ? 'MATCH_PROBABLE' : 'MATCH_REJECTED'}`}>
                                    {statusLabel(cam.status)}
                                </span>
                            </div>

                            {/* Location */}
                            <div style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                {cam.location_name}
                            </div>

                            {/* Stats */}
                            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.65rem' }}>
                                <div>
                                    <div style={{ fontSize: '0.52rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>TRACKS</div>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                                        {tracks.length}
                                    </div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.52rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>FPS</div>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                                        {cam.fps.toFixed(1)}
                                    </div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.52rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>RES</div>
                                    <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                                        {cam.resolution}
                                    </div>
                                </div>
                            </div>

                            {/* Latest Tracks */}
                            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
                                {tracks.length === 0 ? (
                                    <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                                        No active local tracks
                                    </div>
                                ) : (
                                    tracks.slice(0, 3).map(trk => (
                                        <div
                                            key={trk.track_id}
                                            onClick={() => onSelectTrack?.(trk.track_id, trk.camera_id)}
                                            style={{
                                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                padding: '0.3rem 0.4rem', borderRadius: '4px', marginBottom: '0.25rem',
                                                background: 'rgba(0,0,0,0.2)', cursor: 'pointer',
                                                transition: 'background 0.15s',
                                            }}
                                            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(0,212,255,0.08)')}
                                            onMouseLeave={e => (e.currentTarget.style.background = 'rgba(0,0,0,0.2)')}
                                        >
                                            <div>
                                                <span className="identity-label identity-track">{trk.track_id}</span>
                                                <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)', marginLeft: '0.4rem' }}>
                                                    {trk.vehicle_class}
                                                </span>
                                            </div>
                                            <span style={{
                                                fontSize: '0.6rem', fontFamily: 'var(--font-mono)',
                                                color: 'var(--text-muted)',
                                            }}>
                                                {trk.speed_kmh.toFixed(0)} km/h
                                            </span>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
