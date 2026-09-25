import { useState, useEffect, useRef } from 'react';

export function useWebSocket(url: string = 'ws://localhost:8000/ws/traffic') {
    const [isConnected, setIsConnected] = useState<boolean>(false);
    const [lastMessage, setLastMessage] = useState<any>(null);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        let reconnectTimer: any = null;

        function connect() {
            try {
                const ws = new WebSocket(url);
                wsRef.current = ws;

                ws.onopen = () => {
                    setIsConnected(true);
                    console.log('WebSocket connected to ChronoEye Infinity backend');
                };

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        setLastMessage(data);
                    } catch (e) {
                        console.error('Error parsing WebSocket message:', e);
                    }
                };

                ws.onerror = (error) => {
                    console.warn('WebSocket error:', error);
                    setIsConnected(false);
                };

                ws.onclose = () => {
                    setIsConnected(false);
                    // Auto-reconnect after 3 seconds
                    reconnectTimer = setTimeout(() => {
                        console.log('Attempting WebSocket reconnection...');
                        connect();
                    }, 3000);
                };
            } catch (err) {
                setIsConnected(false);
                reconnectTimer = setTimeout(connect, 3000);
            }
        }

        connect();

        return () => {
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (wsRef.current) wsRef.current.close();
        };
    }, [url]);

    return { isConnected, lastMessage };
}
