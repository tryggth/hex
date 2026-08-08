'use strict';

// --- HEX NEIGHBOR DIRECTIONS (offset coordinates) ---
const HEX_DIRS = [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]];

// Colors
const COLOR_BG = '#0d1117';
const COLOR_EMPTY_FILL = '#1a2332';
const COLOR_EMPTY_STROKE = '#2a3a52';
const COLOR_RED = '#FF4444';
const COLOR_BLUE = '#4488FF';

// Player IDs
const EMPTY = 0;
const RED = 1;  // Human, connects Top-to-Bottom
const BLUE = 2; // AI, connects Left-to-Right

// --- UI ELEMENTS ---
const canvas = document.getElementById('hexCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const statusText = document.getElementById('statusText');
const headerSubtitle = document.getElementById('headerSubtitle');

const boardSizeSlider = document.getElementById('boardSizeSlider');
const boardSizeValue = document.getElementById('boardSizeValue');

const timeSlider = document.getElementById('timeSlider');
const timeValue = document.getElementById('timeValue');

const cSlider = document.getElementById('cSlider');
const cValue = document.getElementById('cValue');

const restartBtn = document.getElementById('restartBtn');
const pauseBtn = document.getElementById('pauseBtn');
const stopBtn = document.getElementById('stopBtn');
const installBtn = document.getElementById('installBtn');

const cosmicPanel = document.getElementById('cosmicPanel');
const mctsNodesCount = document.getElementById('mctsNodesCount');
const cosmicPathsCount = document.getElementById('cosmicPathsCount');
const cosmicScaleText = document.getElementById('cosmicScaleText');

// Dynamic geometry state
let currentBoardSize = 7;
let hexSize = 28;
let hexW = Math.sqrt(3) * hexSize;
let hexH = 2 * hexSize;
let offsetX = 60;
let offsetY = 60;

function updateLayoutGeometry(boardSize) {
    currentBoardSize = boardSize;
    if (!canvas) return;
    let availW = canvas.width - 40;  // 540
    let availH = canvas.height - 40; // 480
    
    let totalWFactor = Math.sqrt(3) * (1.5 * boardSize - 0.5);
    let totalHFactor = 1.5 * boardSize + 0.5;
    
    let sizeW = availW / totalWFactor;
    let sizeH = availH / totalHFactor;
    hexSize = Math.min(sizeW, sizeH);
    
    hexW = Math.sqrt(3) * hexSize;
    hexH = 2 * hexSize;
    
    let boardPixelW = hexW * (1.5 * boardSize - 0.5);
    let boardPixelH = (1.5 * boardSize + 0.5) * hexSize;
    
    offsetX = (canvas.width - boardPixelW) / 2 + hexW / 2;
    offsetY = (canvas.height - boardPixelH) / 2 + hexSize;
}

function updateButtonStates() {
    let isAITurn = mcts ? mcts.isRunning : false;
    if (pauseBtn) pauseBtn.disabled = !isAITurn;
    if (stopBtn) stopBtn.disabled = !isAITurn;
}

// Slider event listeners
if (boardSizeSlider) {
    boardSizeSlider.addEventListener('input', () => {
        let size = parseInt(boardSizeSlider.value);
        if (boardSizeValue) boardSizeValue.textContent = `${size}×${size}`;
        if (headerSubtitle) headerSubtitle.textContent = `Human (Red) vs AI (Blue) · ${size}×${size}`;
        if (!mcts.isRunning) {
            resetGame(size);
        }
    });
}

if (timeSlider) {
    timeSlider.addEventListener('input', () => {
        let ms = parseInt(timeSlider.value);
        if (timeValue) timeValue.textContent = `${ms}ms`;
    });
}

if (cSlider) {
    cSlider.addEventListener('input', () => {
        if (cValue) cValue.textContent = parseFloat(cSlider.value).toFixed(2);
    });
}

// --- MATH & COSMIC SCALE HELPERS ---

/**
 * Calculates E! (E factorial) of remaining empty cells using base-10 logarithms to prevent overflow.
 * @param {number} emptyCells - Number of empty cells remaining on the board (E)
 * @returns {object} { formatted: string, exponent: number, mantissa: number }
 */
function calculatePaths(emptyCells) {
    if (emptyCells <= 0) return { formatted: "1", exponent: 0, mantissa: 1 };
    let sumLog10 = 0;
    for (let i = 1; i <= emptyCells; i++) {
        sumLog10 += Math.log10(i);
    }
    let exponent = Math.floor(sumLog10);
    let mantissa = Math.pow(10, sumLog10 - exponent);

    // Map digits to unicode superscripts for scientific notation display
    let superscripts = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'
    };
    let expStr = exponent.toString().split('').map(char => superscripts[char] || char).join('');
    let formatted = `${mantissa.toFixed(2)} × 10${expStr}`;

    return { formatted, exponent, mantissa };
}

/**
 * Returns a universe comparison string based on the base-10 exponent of E!.
 * @param {number} exponent - The base-10 exponent
 * @returns {string} Universe comparison string
 */
function getCosmicScale(exponent) {
    if (exponent >= 185) return "More than the Planck volumes in the observable universe.";
    if (exponent >= 120) return "More than the Shannon Number (all possible Chess games).";
    if (exponent >= 80) return "More than the atoms in the observable universe.";
    if (exponent >= 67) return "More than the atoms in the Milky Way galaxy.";
    if (exponent >= 50) return "More than the atoms making up the Earth.";
    if (exponent >= 25) return "More than the drops of water in all Earth's oceans.";
    if (exponent >= 18) return "More than all the grains of sand on Earth.";
    if (exponent >= 11) return "More than the stars in our galaxy.";
    if (exponent >= 9)  return "More than the human population of Earth.";
    return "A number a standard computer could brute-force.";
}

function updateCosmicPanel() {
    let emptyCells = gameState.getLegalMoves().length;
    let pathInfo = calculatePaths(emptyCells);
    let scaleStr = getCosmicScale(pathInfo.exponent);

    let nodes = (mcts.root && mcts.root.visits) ? mcts.root.visits : 0;

    if (mctsNodesCount) mctsNodesCount.textContent = nodes.toLocaleString();
    if (cosmicPathsCount) cosmicPathsCount.textContent = pathInfo.formatted;
    if (cosmicScaleText) cosmicScaleText.textContent = scaleStr;
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
    constructor(size = 7) {
        this.size = size;
        this.numCells = size * size;
        this.redTop = this.numCells;
        this.redBottom = this.numCells + 1;
        this.blueLeft = this.numCells + 2;
        this.blueRight = this.numCells + 3;
        
        this.board = new Uint8Array(this.numCells);
        this.uf = new UnionFind(this.numCells + 4);
        
        this.currentPlayer = RED;
        this.winner = EMPTY;
        this.movesMade = 0;
    }

    clone() {
        let copy = new HexBoard(this.size);
        copy.board.set(this.board);
        copy.uf.parent.set(this.uf.parent);
        copy.uf.rank.set(this.uf.rank);
        copy.currentPlayer = this.currentPlayer;
        copy.winner = this.winner;
        copy.movesMade = this.movesMade;
        return copy;
    }

    coordToIdx(r, c) {
        return r * this.size + c;
    }

    isValid(r, c) {
        return r >= 0 && r < this.size && c >= 0 && c < this.size;
    }

    getWinner() {
        if (this.uf.find(this.redTop) === this.uf.find(this.redBottom)) return RED;
        if (this.uf.find(this.blueLeft) === this.uf.find(this.blueRight)) return BLUE;
        return EMPTY;
    }

    play(idx) {
        if (this.board[idx] !== EMPTY || this.winner !== EMPTY) return false;
        
        let r = Math.floor(idx / this.size);
        let c = idx % this.size;
        let player = this.currentPlayer;
        
        this.board[idx] = player;
        this.movesMade++;
        
        // Connect to same-color neighbors
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
        
        // Connect border cells to sentinels
        if (player === RED) {
            if (r === 0) this.uf.union(idx, this.redTop);
            if (r === this.size - 1) this.uf.union(idx, this.redBottom);
        } else {
            if (c === 0) this.uf.union(idx, this.blueLeft);
            if (c === this.size - 1) this.uf.union(idx, this.blueRight);
        }
        
        this.winner = this.getWinner();
        this.currentPlayer = player === RED ? BLUE : RED;
        return true;
    }
    
    getLegalMoves() {
        let moves = [];
        for (let i = 0; i < this.numCells; i++) {
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
        return this.state.winner !== EMPTY || this.state.movesMade === this.state.numCells;
    }

    isFullyExpanded() {
        return this.untriedMoves.length === 0;
    }
    
    expand() {
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
    constructor() {
        this.root = null;
        this.isRunning = false;
        this.isPaused = false;
        this.isAborted = false;
        this.accumulatedTime = 0;
    }

    async search(state, timeLimitMs) {
        this.root = new MCTSNode(state);
        this.isRunning = true;
        this.isPaused = false;
        this.isAborted = false;
        this.accumulatedTime = 0;
        let cParam = cSlider ? parseFloat(cSlider.value) : 1.4;
        let lastTime = performance.now();
        
        updateButtonStates();

        return new Promise((resolve) => {
            let iter = () => {
                let now = performance.now();
                if (!this.isPaused) {
                    this.accumulatedTime += (now - lastTime);
                }
                lastTime = now;

                // Stop strategizing / Abort OR time budget reached
                if (this.isAborted || this.accumulatedTime >= timeLimitMs) {
                    this.isRunning = false;
                    this.isPaused = false;
                    this.isAborted = false;
                    updateButtonStates();

                    let bestNode = null;
                    let maxVisits = -1;
                    if (this.root && this.root.children) {
                        for (let child of this.root.children) {
                            if (child.visits > maxVisits) {
                                maxVisits = child.visits;
                                bestNode = child;
                            }
                        }
                    }
                    resolve(bestNode ? bestNode.move : -1);
                    return;
                }

                if (this.isPaused) {
                    // Halts iterations, keeps heatmap frozen, updates panel
                    render(state, this.root);
                    updateCosmicPanel();
                    setTimeout(iter, 100);
                    return;
                }
                
                for (let i = 0; i < 80; i++) {
                    let node = this.select(this.root, cParam);
                    let winner = this.simulate(node.state);
                    this.backpropagate(node, winner);
                }
                
                render(state, this.root);
                updateCosmicPanel();
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
        
        let simState = state.clone();
        let empties = simState.getLegalMoves();
        let size = simState.size;
        let numCells = simState.numCells;
        let redTop = numCells;
        let redBottom = numCells + 1;
        let blueLeft = numCells + 2;
        let blueRight = numCells + 3;
        
        // Fisher-Yates shuffle
        for (let i = empties.length - 1; i > 0; i--) {
            let j = Math.floor(Math.random() * (i + 1));
            let temp = empties[i];
            empties[i] = empties[j];
            empties[j] = temp;
        }
        
        let cur = simState.currentPlayer;
        for (let move of empties) {
            simState.board[move] = cur;
            cur = cur === RED ? BLUE : RED;
        }
        
        let uf = new UnionFind(numCells + 4);
        
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                let idx = r * size + c;
                let p = simState.board[idx];
                
                for (let dir of HEX_DIRS) {
                    let nr = r + dir[0];
                    let nc = c + dir[1];
                    if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                        let nidx = nr * size + nc;
                        if (simState.board[nidx] === p) {
                            uf.union(idx, nidx);
                        }
                    }
                }
                
                if (p === RED) {
                    if (r === 0) uf.union(idx, redTop);
                    if (r === size - 1) uf.union(idx, redBottom);
                } else if (p === BLUE) {
                    if (c === 0) uf.union(idx, blueLeft);
                    if (c === size - 1) uf.union(idx, blueRight);
                }
            }
        }
        
        if (uf.find(redTop) === uf.find(redBottom)) return RED;
        if (uf.find(blueLeft) === uf.find(blueRight)) return BLUE;
        return EMPTY;
    }

    backpropagate(node, winner) {
        while (node !== null) {
            node.visits++;
            if (node.parent) {
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
    let x = offsetX + hexW * c + (r * hexW) / 2;
    let y = offsetY + (r * 3 / 4) * hexH;
    return {x, y};
}

function drawHex(x, y, fillStyle, strokeStyle, lineWidth = 1) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        let angle = Math.PI / 180 * (60 * i - 30);
        let px = x + hexSize * Math.cos(angle);
        let py = y + hexSize * Math.sin(angle);
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
    
    let size = state.size;
    updateLayoutGeometry(size);
    
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

    // Font size scaling based on hexSize
    let numFontSize = Math.max(9, Math.floor(hexSize * 0.32));
    let mctsFontSize = Math.max(10, Math.floor(hexSize * 0.38));

    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            let idx = state.coordToIdx(r, c);
            let player = state.board[idx];
            let {x, y} = getHexCenter(r, c);
            let cellNum = idx + 1; // 1-based hex number
            
            let fill = COLOR_EMPTY_FILL;
            if (player === RED) fill = COLOR_RED;
            else if (player === BLUE) fill = COLOR_BLUE;
            
            let stroke = COLOR_EMPTY_STROKE;
            let lw = 1;
            
            let isThinkingWithVisit = (player === EMPTY && maxVisits > 0 && visitMap[idx] !== undefined);
            let t = 0;
            let visits = 0;

            if (player === EMPTY) {
                let isTopOrBottom = r === 0 || r === size - 1;
                let isLeftOrRight = c === 0 || c === size - 1;
                
                if (isTopOrBottom) {
                    stroke = COLOR_RED;
                    lw = 2;
                }
                if (isLeftOrRight) {
                    if (isTopOrBottom) {
                        stroke = '#AA44AA';
                    } else {
                        stroke = COLOR_BLUE;
                        lw = 2;
                    }
                }

                // Heatmap interior scaling: scale fill from white to cyan/blue
                if (isThinkingWithVisit) {
                    visits = visitMap[idx];
                    t = Math.pow(visits / maxVisits, 0.5);
                    // Scale interior fill from white (#FFFFFF) to final cyan/blue placement color (#0088FF)
                    fill = lerpColor('#FFFFFF', '#0088FF', t);
                    
                    if (!isTopOrBottom && !isLeftOrRight) {
                        stroke = lerpColor('#2a3a52', '#00FFFF', t);
                        lw = 1 + t * 1.5;
                    }
                }
            }

            drawHex(x, y, fill, stroke, lw);

            // Render Cell Text & Heatmap Visit Counts
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            if (player === EMPTY) {
                if (isThinkingWithVisit) {
                    // High-contrast text color based on interior brightness
                    let mainTextColor = t > 0.65 ? '#FFFFFF' : '#0a0e1a';
                    let visitTextColor = t > 0.65 ? '#00FFFF' : '#003366';

                    ctx.fillStyle = mainTextColor;
                    ctx.font = `bold ${numFontSize}px sans-serif`;
                    let numY = y - hexSize * 0.42;
                    ctx.fillText(cellNum.toString(), x, numY);

                    // Heatmap visit count (centered below cell number)
                    ctx.fillStyle = visitTextColor;
                    ctx.font = `bold ${mctsFontSize}px sans-serif`;
                    ctx.fillText(visits.toString(), x, y + hexSize * 0.18);
                } else {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
                    ctx.font = `${numFontSize}px sans-serif`;
                    ctx.fillText(cellNum.toString(), x, y);
                }
            } else {
                // Stone is placed — draw subtle cell number
                ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
                ctx.font = `bold ${numFontSize}px sans-serif`;
                ctx.fillText(cellNum.toString(), x, y);
            }
        }
    }
}

// --- GAME LOOP & STATE ---
let initialSize = boardSizeSlider ? parseInt(boardSizeSlider.value) : 7;
let gameState = new HexBoard(initialSize);
let mcts = new MCTS();

function resetGame(size = currentBoardSize) {
    mcts.isPaused = false;
    mcts.isAborted = false;
    mcts.isRunning = false;
    if (pauseBtn) {
        pauseBtn.textContent = '⏸️ Pause AI';
        pauseBtn.classList.remove('paused');
    }
    if (cosmicPanel) cosmicPanel.style.display = 'none';
    updateButtonStates();

    gameState = new HexBoard(size);
    if (statusText) {
        statusText.textContent = 'Your turn — place a Red stone';
        statusText.className = 'status-text';
    }
    render(gameState);
}

function getEventHex(e) {
    let rect = canvas.getBoundingClientRect();
    let clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    let px = clientX - rect.left;
    let py = clientY - rect.top;
    
    let bestDist = Infinity;
    let bestIdx = -1;
    let size = gameState.size;
    
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            let {x, y} = getHexCenter(r, c);
            let dx = px - x;
            let dy = py - y;
            let dist = dx * dx + dy * dy;
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = gameState.coordToIdx(r, c);
            }
        }
    }
    
    if (bestDist <= hexSize * hexSize) {
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
        updateButtonStates();
        return;
    }
    
    // AI play
    if (statusText) {
        statusText.textContent = mcts.isPaused ? '⏸️ AI Paused — Inspecting MCTS Search' : '🔵 Blue AI thinking...';
        statusText.className = mcts.isPaused ? 'status-text paused' : 'status-text thinking';
    }
    
    let durationMs = timeSlider ? parseInt(timeSlider.value) : 1500;
    let aiMove = await mcts.search(gameState, durationMs);
    
    // Process AI Move
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
    updateButtonStates();
}

// --- EVENT LISTENERS ---
if (canvas) {
    canvas.addEventListener('click', (e) => {
        if (mcts.isRunning && !mcts.isPaused) return;
        let idx = getEventHex(e);
        handleMove(idx);
    });

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        if (mcts.isRunning && !mcts.isPaused) return;
        let idx = getEventHex(e);
        handleMove(idx);
    }, {passive: false});
}

if (restartBtn) {
    restartBtn.addEventListener('click', () => {
        resetGame(currentBoardSize);
    });
}

if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
        if (!mcts.isRunning) return;
        mcts.isPaused = !mcts.isPaused;
        if (mcts.isPaused) {
            pauseBtn.textContent = '▶️ Resume AI';
            pauseBtn.classList.add('paused');
            if (cosmicPanel) cosmicPanel.style.display = 'flex';
            if (statusText) {
                statusText.textContent = '⏸️ AI Paused — Inspecting MCTS Search';
                statusText.className = 'status-text paused';
            }
            updateCosmicPanel();
        } else {
            pauseBtn.textContent = '⏸️ Pause AI';
            pauseBtn.classList.remove('paused');
            if (cosmicPanel) cosmicPanel.style.display = 'none';
            if (statusText && mcts.isRunning) {
                statusText.textContent = '🔵 Blue AI thinking...';
                statusText.className = 'status-text thinking';
            } else if (statusText) {
                statusText.textContent = 'Your turn — place a Red stone';
                statusText.className = 'status-text';
            }
        }
    });
}

if (stopBtn) {
    stopBtn.addEventListener('click', () => {
        if (!mcts.isRunning) return;
        mcts.isAborted = true;
        mcts.isPaused = false;
        if (pauseBtn) {
            pauseBtn.textContent = '⏸️ Pause AI';
            pauseBtn.classList.remove('paused');
        }
        if (cosmicPanel) cosmicPanel.style.display = 'none';
    });
}

// --- PWA INSTALLATION HANDLER ---
let deferredPrompt = null;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (installBtn && !window.matchMedia('(display-mode: standalone)').matches) {
        installBtn.style.display = 'flex';
    }
});

if (installBtn) {
    installBtn.addEventListener('click', async () => {
        if (!deferredPrompt) return;
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        console.log(`PWA install prompt choice: ${outcome}`);
        deferredPrompt = null;
        installBtn.style.display = 'none';
    });
}

window.addEventListener('appinstalled', () => {
    deferredPrompt = null;
    if (installBtn) installBtn.style.display = 'none';
    console.log('Hex MCTS PWA installed successfully');
});

// --- MODAL CONTROLLER ---
const infoBtn = document.getElementById('infoBtn');
const footerInfoLink = document.getElementById('footerInfoLink');
const infoModal = document.getElementById('infoModal');
const closeModalBtn = document.getElementById('closeModalBtn');

function openModal() {
    if (infoModal) {
        infoModal.style.display = 'flex';
        infoModal.setAttribute('aria-hidden', 'false');
    }
}

function closeModal() {
    if (infoModal) {
        infoModal.style.display = 'none';
        infoModal.setAttribute('aria-hidden', 'true');
    }
}

if (infoBtn) infoBtn.addEventListener('click', openModal);
if (footerInfoLink) {
    footerInfoLink.addEventListener('click', (e) => {
        e.preventDefault();
        openModal();
    });
}
if (closeModalBtn) closeModalBtn.addEventListener('click', closeModal);

if (infoModal) {
    infoModal.addEventListener('click', (e) => {
        if (e.target === infoModal) closeModal();
    });
}

window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && infoModal && infoModal.style.display !== 'none') {
        closeModal();
    }
});

// Initialize UI displays and initial render
if (statusText) statusText.textContent = 'Your turn — place a Red stone';
if (boardSizeSlider && boardSizeValue) boardSizeValue.textContent = `${initialSize}×${initialSize}`;
if (timeSlider && timeValue) timeValue.textContent = `${timeSlider ? timeSlider.value : 1500}ms`;
if (cSlider && cValue) cValue.textContent = parseFloat(cSlider.value).toFixed(2);
updateButtonStates();

render(gameState);
