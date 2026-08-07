'use strict';

// --- CONSTANTS ---
const BOARD_SIZE = 7;
const NUM_CELLS = BOARD_SIZE * BOARD_SIZE;

// Sentinels for Union-Find win detection
const RED_TOP = 49;
const RED_BOTTOM = 50;
const BLUE_LEFT = 51;
const BLUE_RIGHT = 52;

// Hex neighbor directions (offset coordinates)
const HEX_DIRS = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];

// Colors
const COLOR_BG = '#0d1117';
const COLOR_EMPTY_FILL = '#1a2332';
const COLOR_EMPTY_STROKE = '#2a3a52';
const COLOR_RED = '#FF4444';
const COLOR_BLUE = '#4488FF';

// Player IDs
const EMPTY = 0;
const RED = 1; // Human, connects Top-to-Bottom
const BLUE = 2; // AI, connects Left-to-Right

// --- UI ELEMENTS ---
const canvas = document.getElementById('hexCanvas');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('statusText');
const cSlider = document.getElementById('cSlider');
const cValue = document.getElementById('cValue');
const restartBtn = document.getElementById('restartBtn');

// Hex layout parameters (assuming 580x520 canvas)
const HEX_SIZE = 28;
const HEX_W = Math.sqrt(3) * HEX_SIZE;
const HEX_H = 2 * HEX_SIZE;
const OFFSET_X = 60;
const OFFSET_Y = 60;

// Update UI slider value display
if (cSlider) {
    cSlider.addEventListener('input', () => {
        if (cValue) cValue.textContent = parseFloat(cSlider.value).toFixed(2);
    });
}

// --- UNION FIND ---
class UnionFind {
    constructor(size) {
        this.parent = new Int32Array(size);
        this.rank = new Int32Array(size);
        for (let i = 0; i < size; i++) {
            this.parent[i] = i;
            this.rank[i] = 0;
        }
    }

    find(i) {
        let root = i;
        while (root !== this.parent[root]) {
            root = this.parent[root];
        }
        // Path compression
        let curr = i;
        while (curr !== root) {
            let nxt = this.parent[curr];
            this.parent[curr] = root;
            curr = nxt;
        }
        return root;
    }

    union(i, j) {
        let rootI = this.find(i);
        let rootJ = this.find(j);
        if (rootI !== rootJ) {
            // Union by rank
            if (this.rank[rootI] < this.rank[rootJ]) {
                this.parent[rootI] = rootJ;
            } else if (this.rank[rootI] > this.rank[rootJ]) {
                this.parent[rootJ] = rootI;
            } else {
                this.parent[rootJ] = rootI;
                this.rank[rootI]++;
            }
        }
    }
}

// --- HEX BOARD STATE ---
class HexBoard {
    constructor() {
        this.board = new Uint8Array(NUM_CELLS);
        this.uf = new UnionFind(NUM_CELLS + 4); // 49 cells + 4 sentinels
        // Sentinels are connected to border cells only when a matching-color
        // stone is placed there — see play() method.
        this.currentPlayer = RED;
        this.winner = EMPTY;
        this.movesMade = 0;
    }

    clone() {
        let copy = new HexBoard();
        copy.board.set(this.board);
        copy.uf.parent.set(this.uf.parent);
        copy.uf.rank.set(this.uf.rank);
        copy.currentPlayer = this.currentPlayer;
        copy.winner = this.winner;
        copy.movesMade = this.movesMade;
        return copy;
    }

    coordToIdx(r, c) {
        return r * BOARD_SIZE + c;
    }

    isValid(r, c) {
        return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
    }

    getWinner() {
        if (this.uf.find(RED_TOP) === this.uf.find(RED_BOTTOM)) return RED;
        if (this.uf.find(BLUE_LEFT) === this.uf.find(BLUE_RIGHT)) return BLUE;
        return EMPTY;
    }

    play(idx) {
        if (this.board[idx] !== EMPTY || this.winner !== EMPTY) return false;
        
        let r = Math.floor(idx / BOARD_SIZE);
        let c = idx % BOARD_SIZE;
        let player = this.currentPlayer;
        
        this.board[idx] = player;
        this.movesMade++;
        
        // Connect to neighbors of the same color
        for (let dir of HEX_DIRS) {
            let nr = r + dir[0];
            let nc = c + dir[1];
            if (this.isValid(nr, nc)) {
                let nidx = this.coordToIdx(nr, nc);
                if (this.board[nidx] === player) {
                    this.uf.union(idx, nidx);
                }
            }
        }
        
        // Connect to sentinel nodes for border cells
        if (player === RED) {
            if (r === 0) this.uf.union(idx, RED_TOP);
            if (r === BOARD_SIZE - 1) this.uf.union(idx, RED_BOTTOM);
        } else {
            if (c === 0) this.uf.union(idx, BLUE_LEFT);
            if (c === BOARD_SIZE - 1) this.uf.union(idx, BLUE_RIGHT);
        }
        
        this.winner = this.getWinner();
        this.currentPlayer = player === RED ? BLUE : RED;
        return true;
    }
    
    getLegalMoves() {
        let moves = [];
        for (let i = 0; i < NUM_CELLS; i++) {
            if (this.board[i] === EMPTY) moves.push(i);
        }
        return moves;
    }
}

// --- MCTS NODE ---
class MCTSNode {
    constructor(state, move = null, parent = null) {
        this.state = state;
        this.move = move;
        this.parent = parent;
        this.children = [];
        this.untriedMoves = state.getLegalMoves();
        this.visits = 0;
        this.wins = 0;
    }

    isTerminal() {
        return this.state.winner !== EMPTY || this.state.movesMade === NUM_CELLS;
    }

    isFullyExpanded() {
        return this.untriedMoves.length === 0;
    }
    
    expand() {
        // Randomly pick an untried move
        let idx = Math.floor(Math.random() * this.untriedMoves.length);
        let move = this.untriedMoves[idx];
        this.untriedMoves.splice(idx, 1);
        
        let nextState = this.state.clone();
        nextState.play(move);
        
        let child = new MCTSNode(nextState, move, this);
        this.children.push(child);
        return child;
    }
    
    bestChild(cParam) {
        let bestScore = -Infinity;
        let best = null;
        for (let child of this.children) {
            let exploit = child.wins / child.visits;
            let explore = Math.sqrt(Math.log(this.visits) / child.visits);
            let score = exploit + cParam * explore;
            if (score > bestScore) {
                bestScore = score;
                best = child;
            }
        }
        return best;
    }
}

// --- MCTS AGENT ---
class MCTS {
    constructor(timeLimitMs) {
        this.timeLimit = timeLimitMs;
        this.root = null;
        this.isRunning = false;
    }

    async search(state) {
        this.root = new MCTSNode(state);
        this.isRunning = true;
        let cParam = cSlider ? parseFloat(cSlider.value) : 1.414;
        
        let startTime = performance.now();
        
        return new Promise((resolve) => {
            let iter = () => {
                // Check time budget
                if (performance.now() - startTime >= this.timeLimit) {
                    this.isRunning = false;
                    let bestNode = null;
                    let maxVisits = -1;
                    // Pick child with most visits (robustness)
                    for (let child of this.root.children) {
                        if (child.visits > maxVisits) {
                            maxVisits = child.visits;
                            bestNode = child;
                        }
                    }
                    resolve(bestNode ? bestNode.move : -1);
                    return;
                }
                
                // Batch iterations to maintain responsiveness
                for (let i = 0; i < 80; i++) {
                    let node = this.select(this.root, cParam);
                    let winner = this.simulate(node.state);
                    this.backpropagate(node, winner);
                }
                
                // Yield to allow rendering heatmap
                render(state, this.root);
                setTimeout(iter, 0);
            };
            iter();
        });
    }

    select(node, cParam) {
        while (!node.isTerminal()) {
            if (!node.isFullyExpanded()) {
                return node.expand();
            } else {
                node = node.bestChild(cParam);
            }
        }
        return node;
    }

    simulate(state) {
        if (state.winner !== EMPTY) return state.winner;
        
        // Nash Trick: Fast evaluation by completely filling the board
        let simState = state.clone();
        let empties = simState.getLegalMoves();
        
        // Fisher-Yates shuffle
        for (let i = empties.length - 1; i > 0; i--) {
            let j = Math.floor(Math.random() * (i + 1));
            let temp = empties[i];
            empties[i] = empties[j];
            empties[j] = temp;
        }
        
        // Fill alternating player moves
        let cur = simState.currentPlayer;
        for (let move of empties) {
            simState.board[move] = cur;
            cur = cur === RED ? BLUE : RED;
        }
        
        // Evaluate full board with a fresh UnionFind
        let uf = new UnionFind(NUM_CELLS + 4);
        
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                let idx = r * BOARD_SIZE + c;
                let p = simState.board[idx];
                
                // Connect to same-color neighbors
                for (let dir of HEX_DIRS) {
                    let nr = r + dir[0];
                    let nc = c + dir[1];
                    if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                        let nidx = nr * BOARD_SIZE + nc;
                        if (simState.board[nidx] === p) {
                            uf.union(idx, nidx);
                        }
                    }
                }
                
                // Connect border cells to sentinels (only matching color)
                if (p === RED) {
                    if (r === 0) uf.union(idx, RED_TOP);
                    if (r === BOARD_SIZE - 1) uf.union(idx, RED_BOTTOM);
                } else if (p === BLUE) {
                    if (c === 0) uf.union(idx, BLUE_LEFT);
                    if (c === BOARD_SIZE - 1) uf.union(idx, BLUE_RIGHT);
                }
            }
        }
        
        if (uf.find(RED_TOP) === uf.find(RED_BOTTOM)) return RED;
        if (uf.find(BLUE_LEFT) === uf.find(BLUE_RIGHT)) return BLUE;
        return EMPTY; // Should never happen in Hex with a full board
    }

    backpropagate(node, winner) {
        while (node !== null) {
            node.visits++;
            if (node.parent) {
                // The move leading to this node was made by parent's currentPlayer
                let playerWhoMoved = node.parent.state.currentPlayer;
                if (playerWhoMoved === winner) {
                    node.wins++;
                }
            }
            node = node.parent;
        }
    }
}

// --- RENDERING ---
function getHexCenter(r, c) {
    // Parallelogram layout offset right by hex_w / 2 per row
    let x = OFFSET_X + HEX_W * c + (r * HEX_W) / 2;
    let y = OFFSET_Y + (r * 3 / 4) * HEX_H;
    return {x, y};
}

function drawHex(x, y, fillStyle, strokeStyle, lineWidth = 1) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        // Pointy-top orientation angles
        let angle = Math.PI / 180 * (60 * i - 30);
        let px = x + HEX_SIZE * Math.cos(angle);
        let py = y + HEX_SIZE * Math.sin(angle);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;
    ctx.stroke();
}

function lerpColor(c1, c2, t) {
    let r1 = parseInt(c1.substring(1, 3), 16);
    let g1 = parseInt(c1.substring(3, 5), 16);
    let b1 = parseInt(c1.substring(5, 7), 16);
    let r2 = parseInt(c2.substring(1, 3), 16);
    let g2 = parseInt(c2.substring(3, 5), 16);
    let b2 = parseInt(c2.substring(5, 7), 16);
    let r = Math.round(r1 + (r2 - r1) * t);
    let g = Math.round(g1 + (g2 - g1) * t);
    let b = Math.round(b1 + (b2 - b1) * t);
    return `#${(1 << 24 | r << 16 | g << 8 | b).toString(16).slice(1).padStart(6, '0')}`;
}

function render(state, rootNode = null) {
    if (!ctx) return;
    
    // Clear background
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let maxVisits = 0;
    let visitMap = {};
    if (rootNode) {
        for (let child of rootNode.children) {
            visitMap[child.move] = child.visits;
            if (child.visits > maxVisits) maxVisits = child.visits;
        }
    }

    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            let idx = state.coordToIdx(r, c);
            let player = state.board[idx];
            let {x, y} = getHexCenter(r, c);
            
            let fill = COLOR_EMPTY_FILL;
            if (player === RED) fill = COLOR_RED;
            else if (player === BLUE) fill = COLOR_BLUE;
            
            let stroke = COLOR_EMPTY_STROKE;
            let lw = 1;

            if (player === EMPTY) {
                // Colored border edges for visual cues
                let isTopOrBottom = r === 0 || r === BOARD_SIZE - 1;
                let isLeftOrRight = c === 0 || c === BOARD_SIZE - 1;
                
                if (isTopOrBottom) {
                    stroke = COLOR_RED;
                    lw = 2;
                }
                if (isLeftOrRight) {
                    if (isTopOrBottom) {
                        // Corners blend both colors
                        stroke = '#AA44AA';
                    } else {
                        stroke = COLOR_BLUE;
                        lw = 2;
                    }
                }
            }

            drawHex(x, y, fill, stroke, lw);

            // Heatmap rendering during AI thinking
            if (player === EMPTY && maxVisits > 0 && visitMap[idx]) {
                let v = visitMap[idx];
                // Non-linear scaling for better heatmap contrast
                let t = Math.pow(v / maxVisits, 0.5); 
                let color = lerpColor('#1a3a5a', '#00FFFF', t);
                
                ctx.fillStyle = color;
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(v.toString(), x, y);
            }
        }
    }
}

// --- GAME LOOP ---
let gameState = new HexBoard();
let mcts = new MCTS(1500); // 1500ms time budget

function getEventHex(e) {
    let rect = canvas.getBoundingClientRect();
    let clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    let px = clientX - rect.left;
    let py = clientY - rect.top;
    
    let bestDist = Infinity;
    let bestIdx = -1;
    
    // Find nearest hex center
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            let {x, y} = getHexCenter(r, c);
            let dx = px - x;
            let dy = py - y;
            let dist = dx*dx + dy*dy;
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = gameState.coordToIdx(r, c);
            }
        }
    }
    
    // Threshold to confirm it's inside the hex
    if (bestDist <= HEX_SIZE * HEX_SIZE) {
        return bestIdx;
    }
    return -1;
}

async function handleMove(idx) {
    if (idx === -1 || gameState.winner !== EMPTY || gameState.currentPlayer !== RED) return;
    if (gameState.board[idx] !== EMPTY) return;
    
    // Human play
    gameState.play(idx);
    render(gameState);
    
    if (gameState.winner === RED) {
        if (statusText) {
            statusText.textContent = '🔴 Red wins!';
            statusText.className = 'status-text win-red';
        }
        return;
    }
    
    // AI play
    if (statusText) {
        statusText.textContent = '🔵 Blue AI thinking...';
        statusText.className = 'status-text thinking';
    }
    
    let aiMove = await mcts.search(gameState);
    if (aiMove !== -1) {
        gameState.play(aiMove);
    }
    render(gameState);
    
    if (gameState.winner === BLUE) {
        if (statusText) {
            statusText.textContent = '🔵 Blue wins!';
            statusText.className = 'status-text win-blue';
        }
    } else {
        if (statusText) {
            statusText.textContent = 'Your turn — place a Red stone';
            statusText.className = 'status-text';
        }
    }
}

// --- EVENT LISTENERS ---
if (canvas) {
    canvas.addEventListener('click', (e) => {
        if (mcts.isRunning) return;
        let idx = getEventHex(e);
        handleMove(idx);
    });

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        if (mcts.isRunning) return;
        let idx = getEventHex(e);
        handleMove(idx);
    }, {passive: false});
}

if (restartBtn) {
    restartBtn.addEventListener('click', () => {
        if (mcts.isRunning) return;
        gameState = new HexBoard();
        if (statusText) {
            statusText.textContent = 'Your turn — place a Red stone';
            statusText.className = 'status-text';
        }
        render(gameState);
    });
}

// Initialize
if (statusText) statusText.textContent = 'Your turn — place a Red stone';
if (cSlider && cValue) cValue.textContent = parseFloat(cSlider.value).toFixed(2);
render(gameState);
