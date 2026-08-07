/*
 * Service Worker for Hex MCTS PWA
 * ================================
 * Cache-first strategy with self-updating flow.
 * To deploy an update: increment CACHE_VERSION and push new files.
 * The new SW will install, skipWaiting, activate (purge old cache),
 * claim clients, and trigger a controllerchange → page reload.
 */

const CACHE_VERSION = 'v6';
const CACHE_NAME = `hex-mcts-${CACHE_VERSION}`;

// All assets to pre-cache on install
const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './style.css',
    './app.js',
    './manifest.json',
    './icons/icon-192x192.png',
    './icons/icon-512x512.png'
];

// ── Install: cache all assets, then skip waiting ──
self.addEventListener('install', (event) => {
    console.log(`[SW] Installing ${CACHE_NAME}`);
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Pre-caching assets');
                return cache.addAll(ASSETS_TO_CACHE);
            })
            .then(() => {
                // Force this SW to become active immediately
                return self.skipWaiting();
            })
    );
});

// ── Activate: delete old caches, then claim all clients ──
self.addEventListener('activate', (event) => {
    console.log(`[SW] Activating ${CACHE_NAME}`);
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== CACHE_NAME)
                        .map((name) => {
                            console.log(`[SW] Deleting old cache: ${name}`);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => {
                // Take control of all open tabs immediately
                return self.clients.claim();
            })
    );
});

// ── Fetch: cache-first, fallback to network ──
self.addEventListener('fetch', (event) => {
    // Only handle same-origin GET requests
    if (event.request.method !== 'GET') return;

    event.respondWith(
        caches.match(event.request)
            .then((cachedResponse) => {
                if (cachedResponse) {
                    return cachedResponse;
                }
                // Not in cache — fetch from network and cache dynamically
                return fetch(event.request).then((networkResponse) => {
                    // Only cache successful same-origin responses
                    if (networkResponse && networkResponse.status === 200 &&
                        networkResponse.type === 'basic') {
                        const responseClone = networkResponse.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return networkResponse;
                });
            })
            .catch(() => {
                // Offline fallback — serve index for navigation requests
                if (event.request.mode === 'navigate') {
                    return caches.match('./index.html');
                }
            })
    );
});
