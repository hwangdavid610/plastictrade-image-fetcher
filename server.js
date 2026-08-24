require("dotenv").config();

const express = require("express");
const OpenAI = require("openai");
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

let cv = null;

try {
    cv = require("@u4/opencv4nodejs");
    console.log("OpenCV enabled");
} catch (error) {
    console.warn(
        "OpenCV unavailable — running Vision-only mode:",
        error.message
    );
}

const app = express();

const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

if (!OPENAI_API_KEY) {
    console.warn(
        "Warning: OPENAI_API_KEY is not set. /process will fail until it is configured."
    );
}

const openai = new OpenAI({
    apiKey: OPENAI_API_KEY
});

app.use(express.json({ limit: "20mb" }));

app.get("/health", (_req, res) => {
    res.json({ ok: true });
});

const TEMPLATE_PATH = path.join(__dirname, "new_template.jpg");
const TARGET_W = 1116;
const TARGET_H = 722;

const KILOGRAM_TEMPLATE_PATH = path.join(
    __dirname,
    "kilogram_template.png"
);
const KILOGRAM_W = 313;
const KILOGRAM_H = 733;

// LOGÍSTICA radio buttons on aligned new_template.jpg
const logisticas = [
    "TAGA",
    "SERRANO",
    "JUAN CARLOS",
    "TREESEVER",
    "ROSSET",
    "PLASTIC",
    "BIOAMBIENTALISTIK"
];

const logisticas_cp = [
    [84, 232],
    [221, 236],
    [347, 236],
    [496, 236],
    [635, 235],
    [749, 237],
    [868, 236]
];

// UNIDAD DE TRANSPORTE radio buttons (right column)
const unidads = [
    "Caja seca",
    "Tolva 30m3",
    "Remolque",
    "Torthon",
    "Cartucho",
    "Olla 17m3",
    "Camioneta",
    "Tolva 7m3",
    "Contenedores CGR"
];

const unidads_cp = [
    [936, 293],
    [934, 318],
    [938, 344],
    [936, 370],
    [938, 395],
    [934, 421],
    [938, 447],
    [937, 475],
    [939, 499]
];

// Value boxes on aligned new_template.jpg
const fieldClips = {
    // Keep these tight so labels / logistics text do not leak in.
    sitio: [705, 112, 885, 148],
    folio: [890, 112, 1095, 148],
    fecha: [705, 170, 1095, 208],

    hora_entrada: [20, 544, 293, 580],
    hora_salida: [298, 544, 564, 576],
    id_operador: [570, 541, 770, 574],
    nombre: [774, 540, 1086, 573],
    placas_vehiculo: [23, 602, 382, 632],
    placas_caja_remolque: [386, 597, 736, 627],
    numero_marchamo: [741, 592, 1091, 623],

    elaboro: [21, 664, 290, 695],
    supervisor: [297, 661, 560, 692],
    autorizo: [568, 657, 828, 690],
    operador_firma: [834, 654, 1089, 682]
};

// Whole SITIO / FOLIO / FECHA + LOGÍSTICA strip for a focused Vision pass.
const HEADER_BLOCK = [680, 95, 1110, 250];
const LOGISTICA_STRIP = [20, 215, 1100, 255];

const FIELD_HINTS = {
    sitio: "Extract only the site code in this value box (e.g. MXCD-03 or MXCD06). Ignore labels.",
    folio: "Extract only the folio number. It may be red. Digits preferred (N° 0251 -> 0251).",
    fecha: "Extract only the handwritten date digits in this value box (e.g. 25/08/2026). Ignore LOGÍSTICA text below.",
    hora_entrada: "Extract only the entry date and time.",
    hora_salida: "Extract only the exit date and time.",
    id_operador: "Extract only the operator ID.",
    nombre: "Extract only the operator full name.",
    placas_vehiculo: "Extract only the vehicle license plate. If it says S/N or SN, return S/N.",
    placas_caja_remolque:
        "Extract only the trailer/box license plate text. " +
        "If handwritten S/N, s/n, SN, or Sin Número, return exactly S/N. Do not return empty.",
    numero_marchamo: "Extract only the seal/marchamo number. Digits preferred.",
    elaboro: "Extract only the handwritten name in this signature box.",
    supervisor: "Extract only the handwritten name in this signature box.",
    autorizo: "Extract only the handwritten name in this signature box.",
    operador_firma: "Extract only the handwritten name in this signature box."
};

const LOGISTICA_REL_X = [
    84 / 1116,
    221 / 1116,
    347 / 1116,
    496 / 1116,
    635 / 1116,
    749 / 1116,
    868 / 1116
];

// Approximate UNIDAD DE MEDIDA radio x positions on full-width photos.
const UDM_REL_X = {
    // Circles sit LEFT of each label inside the UDM cell (~0.48–0.72 of page).
    Playo: [
        { name: "A granel", x: 0.504 },
        { name: "Pacas", x: 0.54 },
        { name: "Gaylord's", x: 0.571 },
        { name: "Barcinas", x: 0.62 }
    ],
    Carton: [
        { name: "A granel", x: 0.52 },
        { name: "Gaylord's", x: 0.575 }
    ],
    RSU: [{ name: "A granel", x: 0.52 }],
    Tarima: [{ name: "Piezas", x: 0.55 }],
    "Tubo de carton": [{ name: "Piezas", x: 0.55 }],
    Otro: [
        { name: "A granel", x: 0.505 },
        { name: "Piezas", x: 0.55 },
        { name: "Gaylord's", x: 0.6 }
    ]
};

function isInkPixel(r, g, b) {
    // Prefer blue/purple ballpoint. Ignore gray printed text (low chroma).
    const avg = (r + g + b) / 3;
    const chroma = Math.max(r, g, b) - Math.min(r, g, b);
    const blueBias = b - Math.max(r, g);

    if (avg < 130 && chroma < 12) {
        return false;
    }

    // Clear blue / cyan ballpoint.
    if (b > r + 18 && b > g + 10 && r < 170 && avg < 200) {
        return true;
    }

    // Dark blue / purple fill inside radios.
    if (avg < 120 && blueBias >= -5 && chroma >= 14 && b >= r - 5) {
        return true;
    }

    return false;
}

function scoreInkColumn(data, width, height, channels, x, yStart, yEnd) {
    let score = 0;
    const x0 = Math.max(0, x - 1);
    const x1 = Math.min(width - 1, x + 1);

    for (let y = yStart; y <= yEnd; y++) {
        for (let xx = x0; xx <= x1; xx++) {
            const idx = (y * width + xx) * channels;
            if (isInkPixel(data[idx], data[idx + 1], data[idx + 2])) {
                score += 1;
            }
        }
    }

    return score;
}

function findInkPeaks(colScores, minSep = 28, minScore = 8) {
    const peaks = [];

    for (let x = 2; x < colScores.length - 2; x++) {
        const s = colScores[x];
        if (
            s < minScore ||
            s < colScores[x - 1] ||
            s < colScores[x + 1]
        ) {
            continue;
        }

        if (!peaks.length || x - peaks[peaks.length - 1].x >= minSep) {
            peaks.push({ x, score: s });
        } else if (s > peaks[peaks.length - 1].score) {
            peaks[peaks.length - 1] = { x, score: s };
        }
    }

    return peaks;
}

function nearestOptionIndex(xf, options, maxDist = 0.07) {
    let bestIdx = -1;
    let bestDist = Infinity;

    for (let i = 0; i < options.length; i++) {
        const dist = Math.abs(options[i].x - xf);
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }

    if (bestIdx < 0 || bestDist > maxDist) {
        return -1;
    }

    return bestIdx;
}

async function detectChoiceByInkPeaks(
    imageBuffer,
    options,
    {
        yStart = 0.55,
        yEnd = 0.95,
        minPeak = 10,
        minScore = 25,
        minMargin = 8,
        maxX = 0.78,
        // When true, use only the single strongest peak (best for LOGÍSTICA).
        singlePeak = false
    } = {}
) {
    const { data, info } = await sharp(imageBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const channels = info.channels;
    const y0 = Math.max(0, Math.floor(yStart * height));
    const y1 = Math.min(height - 1, Math.floor(yEnd * height));
    const colScores = new Array(width).fill(0);

    for (let x = 0; x < width; x++) {
        colScores[x] = scoreInkColumn(
            data,
            width,
            height,
            channels,
            x,
            y0,
            y1
        );
    }

    const smooth = colScores.map((_, x) => {
        let sum = 0;
        let n = 0;
        for (let dx = -5; dx <= 5; dx++) {
            const xx = x + dx;
            if (xx >= 0 && xx < width) {
                sum += colScores[xx];
                n += 1;
            }
        }
        return sum / n;
    });

    const peaks = findInkPeaks(
        smooth,
        Math.floor(width * 0.045),
        minPeak
    ).filter((p) => {
        const xf = p.x / width;
        return xf >= 0.03 && xf <= maxX;
    });

    if (!peaks.length) {
        return null;
    }

    peaks.sort((a, b) => b.score - a.score);

    if (singlePeak) {
        const top = peaks[0];
        const second = peaks[1];
        if (
            top.score < minScore ||
            (second && top.score - second.score < minMargin)
        ) {
            return null;
        }

        const idx = nearestOptionIndex(top.x / width, options, 0.075);
        if (idx < 0) {
            return null;
        }

        return {
            name: options[idx].name,
            score: top.score,
            margin: top.score - (second?.score || 0),
            scores: options.map((_, i) => (i === idx ? top.score : 0)),
            peaks: peaks.slice(0, 5).map((p) => ({
                x: +(p.x / width).toFixed(3),
                score: +p.score.toFixed(1)
            }))
        };
    }

    const scores = options.map(() => 0);

    for (const peak of peaks) {
        const idx = nearestOptionIndex(peak.x / width, options, 0.07);
        if (idx >= 0) {
            scores[idx] = Math.max(scores[idx], peak.score);
        }
    }

    const ranked = scores
        .map((score, index) => ({
            score,
            index,
            name: options[index].name
        }))
        .sort((a, b) => b.score - a.score);

    const top = ranked[0];
    const second = ranked[1];

    if (
        !top ||
        top.score < minScore ||
        (second && top.score - second.score < minMargin)
    ) {
        return null;
    }

    return {
        name: top.name,
        score: top.score,
        margin: top.score - (second?.score || 0),
        scores,
        peaks: peaks.slice(0, 5).map((p) => ({
            x: +(p.x / width).toFixed(3),
            score: +p.score.toFixed(1)
        }))
    };
}

async function detectChoiceByInk(
    imageBuffer,
    options,
    {
        yStart = 0.55,
        yEnd = 0.92,
        ySteps = 10,
        xSlop = 0.03,
        xSteps = 7,
        minScore = 25,
        minMargin = 10,
        maxX = 0.75,
        radius = 9,
        usePeaks = true,
        singlePeak = false
    } = {}
) {
    if (usePeaks) {
        return detectChoiceByInkPeaks(imageBuffer, options, {
            yStart,
            yEnd,
            minScore,
            minMargin,
            maxX,
            singlePeak
        });
    }

    const { data, info } = await sharp(imageBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const channels = info.channels;
    let best = null;

    for (let step = 0; step < ySteps; step++) {
        const y =
            yStart + ((yEnd - yStart) * step) / Math.max(ySteps - 1, 1);
        const cy = Math.round(y * height);
        const scores = [];

        for (const opt of options) {
            if (opt.x > maxX) {
                scores.push(0);
                continue;
            }

            let optionBest = 0;

            for (let xs = 0; xs < xSteps; xs++) {
                const dx =
                    -xSlop + (2 * xSlop * xs) / Math.max(xSteps - 1, 1);
                const x = Math.min(maxX, Math.max(0.03, opt.x + dx));
                const cx = Math.round(x * width);
                let ink = 0;

                for (let dy = -radius; dy <= radius; dy++) {
                    for (let ddx = -radius; ddx <= radius; ddx++) {
                        if (ddx * ddx + dy * dy > radius * radius) {
                            continue;
                        }
                        const px = cx + ddx;
                        const py = cy + dy;
                        if (
                            px < 0 ||
                            py < 0 ||
                            px >= width ||
                            py >= height
                        ) {
                            continue;
                        }
                        const idx = (py * width + px) * channels;
                        if (
                            isInkPixel(
                                data[idx],
                                data[idx + 1],
                                data[idx + 2]
                            )
                        ) {
                            ink += 1;
                        }
                    }
                }

                if (ink > optionBest) {
                    optionBest = ink;
                }
            }

            scores.push(optionBest);
        }

        const ranked = scores
            .map((score, index) => ({
                score,
                index,
                name: options[index].name
            }))
            .sort((a, b) => b.score - a.score);

        const top = ranked[0];
        const second = ranked[1];

        if (
            top &&
            top.score >= minScore &&
            (!second || top.score - second.score >= minMargin)
        ) {
            if (!best || top.score > best.score) {
                best = {
                    name: top.name,
                    score: top.score,
                    margin: top.score - (second?.score || 0),
                    y,
                    scores
                };
            }
        }
    }

    return best;
}

function paperBrightFrac(data, width, height, channels, x) {
    let bright = 0;
    let n = 0;

    for (
        let y = Math.floor(height * 0.2);
        y < Math.floor(height * 0.85);
        y += 2
    ) {
        const idx = (y * width + x) * channels;
        const avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        n += 1;
        if (avg > 140) {
            bright += 1;
        }
    }

    return n ? bright / n : 0;
}

async function detectPaperLeftFrac(data, width, height, channels) {
    for (let x = 0; x < Math.floor(width * 0.28); x++) {
        if (paperBrightFrac(data, width, height, channels, x) > 0.65) {
            return x / width;
        }
    }

    return 0.05;
}

async function detectPaperRightFrac(data, width, height, channels) {
    for (let x = width - 1; x > Math.floor(width * 0.55); x--) {
        if (paperBrightFrac(data, width, height, channels, x) > 0.65) {
            return x / width;
        }
    }

    return 0.92;
}

function scoreInkDisk(data, width, height, channels, cx, cy, radius) {
    let score = 0;

    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy > radius * radius) {
                continue;
            }

            const x = cx + dx;
            const y = cy + dy;

            if (x < 0 || y < 0 || x >= width || y >= height) {
                continue;
            }

            const idx = (y * width + x) * channels;
            if (isInkPixel(data[idx], data[idx + 1], data[idx + 2])) {
                score += 1;
            }
        }
    }

    return score;
}

async function detectLogisticaByBlueInk(imageBuffer) {
    const { data, info } = await sharp(imageBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const channels = info.channels;
    const paperLeft = await detectPaperLeftFrac(
        data,
        width,
        height,
        channels
    );
    const paperRight = await detectPaperRightFrac(
        data,
        width,
        height,
        channels
    );
    const paperWidth = Math.max(0.5, paperRight - paperLeft);

    // Template radios are fractions of the full form, not the paper edge.
    const options = logisticas.map((name, index) => ({
        name,
        x: paperLeft + LOGISTICA_REL_X[index] * paperWidth
    }));

    const y0 = Math.floor(height * 0.38);
    const y1 = Math.floor(height * 0.82);
    const radius = Math.max(8, Math.round(width * 0.012));
    const xSlop = Math.round(width * 0.018);

    const scores = options.map((opt) => {
        const cx0 = Math.round(opt.x * width);
        let best = 0;

        for (let y = y0; y <= y1; y += 2) {
            for (let dx = -xSlop; dx <= xSlop; dx += 2) {
                const score = scoreInkDisk(
                    data,
                    width,
                    height,
                    channels,
                    cx0 + dx,
                    y,
                    radius
                );
                if (score > best) {
                    best = score;
                }
            }
        }

        return best;
    });

    const ranked = scores
        .map((score, index) => ({
            score,
            index,
            name: options[index].name
        }))
        .sort((a, b) => b.score - a.score);

    const top = ranked[0];
    const second = ranked[1];
    const margin = top ? top.score - (second?.score || 0) : 0;

    if (!top || top.score < 18 || margin < Math.max(6, top.score * 0.12)) {
        return null;
    }

    return {
        selected_logistica: top.name,
        logistica_option: top.index + 1,
        score: top.score,
        margin,
        scores,
        paperLeft,
        paperRight
    };
}

function logisticaIndex(name) {
    const normalized = normalizeLogistica(name);
    return normalized ? logisticas.indexOf(normalized) : -1;
}

function logisticaFromOperadorId(id) {
    const token = String(id || "")
        .toUpperCase()
        .replace(/\s+/g, "");

    if (!token) {
        return null;
    }

    const match = token.match(
        /PT-?(BIO|SR|TH|TG|R[HA]|JC|PT)(?:-?\d)/
    );

    if (!match) {
        return null;
    }

    switch (match[1]) {
        case "BIO":
            return "BIOAMBIENTALISTIK";
        case "SR":
            return "SERRANO";
        case "TH":
            return "TREESEVER";
        case "TG":
            return "TAGA";
        case "RH":
        case "RA":
            return "ROSSET";
        case "JC":
            return "JUAN CARLOS";
        case "PT":
            return "PLASTIC";
        default:
            return null;
    }
}

function pickSelectedLogistica(
    inkResult,
    visionResult,
    fromOperador = null
) {
    const fromMarks = resolveLogistica(visionResult);
    const fromInk = normalizeLogistica(inkResult?.selected_logistica);
    const fromOp = normalizeLogistica(fromOperador);
    const marksIdx = logisticaIndex(fromMarks);
    const inkIdx = logisticaIndex(fromInk);
    const opIdx = logisticaIndex(fromOp);

    // Operator IDs are prefixed by logistics (PT-SR, PT-PT, PT-TH, ...).
    // Use them to confirm Vision, or to correct off-by-one (filled circle
    // attributed to the previous printed name). Do not override a clear
    // unrelated mark (e.g. PLASTIC selected with a PT-TG operator).
    if (fromOp && fromOp === fromMarks) {
        return fromOp;
    }

    if (fromOp) {
        if (marksIdx >= 0 && opIdx === marksIdx + 1) {
            return fromOp;
        }

        if (inkIdx >= 0 && opIdx === inkIdx + 1) {
            return fromOp;
        }
    }

    // Ink one option to the right of Vision is the same off-by-one, but
    // a false far-right ink hit (BIO) must not beat a confirmed mark.
    if (
        marksIdx >= 0 &&
        inkIdx === marksIdx + 1 &&
        (!fromOp || fromOp === fromInk)
    ) {
        return fromInk;
    }

    const marks = Array.isArray(visionResult?.logistica_marks)
        ? visionResult.logistica_marks
        : Array.isArray(visionResult?.marks)
          ? visionResult.marks
          : null;
    const uniqueFilled = marks
        ? marks.filter((mark) => mark && mark.filled)
        : [];

    if (uniqueFilled.length === 1 && fromMarks) {
        return fromMarks;
    }

    if (fromInk && fromMarks && fromInk === fromMarks) {
        return fromInk;
    }

    if (
        fromInk &&
        inkResult?.score >= 18 &&
        inkResult?.margin >= Math.max(6, inkResult.score * 0.12)
    ) {
        return fromInk;
    }

    return fromMarks || fromInk || fromOp || null;
}

async function extractRowUnidadWithVision(
    rowJpeg,
    materialName,
    optionNames
) {
    if (!OPENAI_API_KEY) {
        return null;
    }

    const base64 = rowJpeg.toString("base64");
    const optionsList = optionNames.join(", ");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 200,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You inspect ONE material row from a Plastic Trade ORDEN DE SALIDA. " +
                    "Return JSON only: {\"unidad\": string|null, \"udm_marks\":[{\"name\":string,\"filled\":boolean}]}.\n" +
                    "Rules:\n" +
                    `- Material row is ${materialName}.\n` +
                    `- UNIDAD DE MEDIDA options in this row (left-to-right): ${optionsList}.\n` +
                    "- Ignore Cantidad handwriting and ignore UNIDAD DE TRANSPORTE on the far right (Caja seca, Tolva, etc).\n" +
                    "- filled=true ONLY if that option's circle has blue/black ink inside.\n" +
                    "- Exactly one option should be filled. unidad = that option name.\n" +
                    "- Empty ring => filled=false."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text:
                            "Which UNIDAD DE MEDIDA circle is filled in this row?"
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        return null;
    }
}

async function refineMaterialsUnidadByBlueInk(
    materials,
    materialsImageBuffer
) {
    if (!Array.isArray(materials) || !materials.length) {
        return materials;
    }

    const rowY = {
        Playo: 0.1,
        Carton: 0.22,
        RSU: 0.34,
        Tarima: 0.46,
        "Tubo de carton": 0.58,
        Organicos: 0.7,
        Chatarra: 0.82,
        Otro: 0.92
    };

    const UDM_CROP_LEFT = 0.48;
    const UDM_CROP_WIDTH = 0.24;

    const meta = await sharp(materialsImageBuffer).metadata();
    const height = meta.height || 1;
    const width = meta.width || 1;
    const refined = [];

    for (const item of materials) {
        const material = normalizeMaterialName(
            item?.material,
            item?.otro_text || item?.otroText
        );
        const baseName = material?.startsWith("Otro")
            ? "Otro"
            : material;
        const udmOptions = UDM_REL_X[baseName];

        if (!udmOptions || udmOptions.length < 2 || !rowY[baseName]) {
            refined.push(item);
            continue;
        }

        const cy = rowY[baseName];
        const bandTop = Math.max(0, Math.floor((cy - 0.06) * height));
        const bandHeight = Math.max(28, Math.floor(0.14 * height));

        // UDM cell only — exclude cantidad digits and transporte radios.
        const udmJpeg = await sharp(materialsImageBuffer)
            .extract({
                left: Math.floor(width * UDM_CROP_LEFT),
                top: bandTop,
                width: Math.floor(width * UDM_CROP_WIDTH),
                height: Math.min(bandHeight, height - bandTop)
            })
            .resize({ width: 900 })
            .jpeg({ quality: 95 })
            .toBuffer();

        const { data, info } = await sharp(udmJpeg)
            .removeAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });

        const options = udmOptions.map((opt) => ({
            name: opt.name,
            x: (opt.x - UDM_CROP_LEFT) / UDM_CROP_WIDTH
        }));

        const colScores = new Array(info.width).fill(0);
        const y0 = Math.floor(info.height * 0.1);
        const y1 = Math.floor(info.height * 0.85);

        for (let x = 0; x < info.width; x++) {
            for (let y = y0; y <= y1; y++) {
                const idx = (y * info.width + x) * info.channels;
                if (
                    isInkPixel(data[idx], data[idx + 1], data[idx + 2])
                ) {
                    colScores[x] += 1;
                }
            }
        }

        const smooth = colScores.map((_, x) => {
            let sum = 0;
            let n = 0;
            for (let dx = -5; dx <= 5; dx++) {
                const xx = x + dx;
                if (xx >= 0 && xx < info.width) {
                    sum += colScores[xx];
                    n += 1;
                }
            }
            return sum / n;
        });

        const peaks = [];
        for (let x = 2; x < info.width - 2; x++) {
            const s = smooth[x];
            if (s < 8 || s < smooth[x - 1] || s < smooth[x + 1]) {
                continue;
            }
            const xf = x / info.width;
            // Keep Caja seca / right-column noise out of UDM peaks.
            if (xf < 0.04 || xf > 0.88) {
                continue;
            }
            if (
                !peaks.length ||
                x - peaks[peaks.length - 1].x >=
                    Math.floor(info.width * 0.1)
            ) {
                peaks.push({ x, score: s });
            } else if (s > peaks[peaks.length - 1].score) {
                peaks[peaks.length - 1] = { x, score: s };
            }
        }

        peaks.sort((a, b) => b.score - a.score);
        let inkUnidad = null;

        if (peaks[0] && peaks[0].score >= 10) {
            const xf = peaks[0].x / info.width;
            let bestIdx = -1;
            let bestDist = Infinity;
            for (let i = 0; i < options.length; i++) {
                const dist = Math.abs(options[i].x - xf);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            if (bestIdx >= 0 && bestDist <= 0.18) {
                inkUnidad = options[bestIdx].name;
            }
        }

        // Fallback: focused Vision on the same UDM crop.
        let visionUnidad = null;
        if (!inkUnidad) {
            try {
                const rowResult = await extractRowUnidadWithVision(
                    udmJpeg,
                    baseName,
                    udmOptions.map((o) => o.name)
                );
                visionUnidad =
                    normalizeUnidadMedida(rowResult?.unidad) ||
                    resolveMaterialUnidad(rowResult);
            } catch (error) {
                console.warn(
                    "Row UDM Vision failed:",
                    baseName,
                    error.message
                );
            }
        }

        const unidad =
            normalizeUnidadMedida(inkUnidad) ||
            visionUnidad ||
            resolveMaterialUnidad(item);

        if (unidad) {
            refined.push({
                ...item,
                unidad,
                udm_marks: udmOptions.map((opt) => ({
                    name: opt.name,
                    filled: normalizeUnidadMedida(opt.name) === unidad
                }))
            });
        } else {
            refined.push(item);
        }
    }

    return refined;
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function cleanOcrText(text) {
    if (text == null) {
        return "";
    }

    let cleaned = String(text)
        .replace(/\r/g, "")
        .replace(/\n+/g, " ")
        .replace(/`+/g, "")
        .replace(/^["'\s]+|["'\s]+$/g, "")
        .replace(/\s+/g, " ")
        .trim();

    // Vision/OCR sometimes apologizes on blank crops instead of returning "".
    if (
        !cleaned ||
        cleaned === '""' ||
        cleaned === "''" ||
        /i'?m sorry/i.test(cleaned) ||
        /can'?t (extract|read|help)/i.test(cleaned) ||
        /unable to (extract|read|determine)/i.test(cleaned) ||
        /no text/i.test(cleaned) ||
        /blank or unreadable/i.test(cleaned) ||
        /^(n\/?a|none|null|unknown)$/i.test(cleaned)
    ) {
        return "";
    }

    return cleaned;
}

function parseInteger(text) {
    if (text == null || text === "") {
        return null;
    }

    if (typeof text === "number" && Number.isFinite(text)) {
        return Math.trunc(text);
    }

    const value = String(text).replace(/[^\d]/g, "");

    if (!value) {
        return null;
    }

    return Number.parseInt(value, 10);
}

function parseWeight(text) {
    if (text == null || text === "") {
        return null;
    }

    if (typeof text === "number" && Number.isFinite(text)) {
        return text;
    }

    let cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    // Keep digits, comma, and dot. Examples: 20,060.00 / 20060.00 / 20.060,00
    cleaned = cleaned.replace(/[^\d.,]/g, "");

    if (!cleaned) {
        return null;
    }

    const hasComma = cleaned.includes(",");
    const hasDot = cleaned.includes(".");

    if (hasComma && hasDot) {
        // Assume the last separator is the decimal mark.
        if (cleaned.lastIndexOf(",") > cleaned.lastIndexOf(".")) {
            cleaned = cleaned.replace(/\./g, "").replace(",", ".");
        } else {
            cleaned = cleaned.replace(/,/g, "");
        }
    } else if (hasComma) {
        // 20060,00 or 20,060
        const parts = cleaned.split(",");

        if (parts.length === 2 && parts[1].length <= 2) {
            cleaned = `${parts[0]}.${parts[1]}`;
        } else {
            cleaned = cleaned.replace(/,/g, "");
        }
    }

    const value = Number.parseFloat(cleaned);

    return Number.isFinite(value) ? value : null;
}

function parseDate(text) {
    text = cleanOcrText(text);

    if (!text) {
        return null;
    }

    let match = text.match(/(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})/);

    if (match) {
        const [, y, m, d] = match;
        return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
    }

    match = text.match(/(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})/);

    if (match) {
        const [, d, m, y] = match;
        return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
    }

    return text;
}

function parseDateTime(text) {
    text = cleanOcrText(text);

    if (!text) {
        return null;
    }

    const match = text.match(
        /(\d{1,2})[-\/](\d{1,2})[-\/](\d{2,4}).*?(\d{1,2}):(\d{2})(?::(\d{2}))?/
    );

    if (!match) {
        return text;
    }

    let [, d, m, y, hour, minute, second] = match;

    if (y.length === 2) {
        y = `20${y}`;
    }

    const base =
        `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")} ` +
        `${hour.padStart(2, "0")}:${minute}`;

    return second != null ? `${base}:${second}` : base;
}

function parseTimeOnly(text) {
    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    const match = cleaned.match(/\b(\d{1,2}):(\d{2})\b/);

    if (!match) {
        return null;
    }

    let hour = Number.parseInt(match[1], 10);
    const minute = match[2];

    if (hour > 23) {
        return null;
    }

    return `${String(hour).padStart(2, "0")}:${minute}`;
}

function parseHorario(text, fechaFallback = null) {
    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    const time = parseTimeOnly(cleaned);
    let date = null;

    // Prefer an explicit date in the field, but reject nonsense years.
    const parsed = parseDateTime(cleaned);

    if (parsed && /^\d{4}-\d{2}-\d{2}/.test(parsed)) {
        const year = Number.parseInt(parsed.slice(0, 4), 10);
        const fallbackYear = fechaFallback
            ? Number.parseInt(String(fechaFallback).slice(0, 4), 10)
            : null;

        if (
            year >= 2020 &&
            year <= 2035 &&
            (fallbackYear == null || Math.abs(year - fallbackYear) <= 1)
        ) {
            date = parsed.slice(0, 10);
        }
    }

    if (!date && fechaFallback && /^\d{4}-\d{2}-\d{2}/.test(fechaFallback)) {
        date = fechaFallback.slice(0, 10);
    }

    if (date && time) {
        return `${date} ${time}`;
    }

    if (time) {
        return time;
    }

    if (date) {
        return date;
    }

    return null;
}

function resolveLogistica(rawOrText, option = null) {
    const source =
        typeof rawOrText === "object" && rawOrText != null
            ? rawOrText
            : null;

    const marks = Array.isArray(source?.logistica_marks)
        ? source.logistica_marks
        : Array.isArray(source?.marks)
          ? source.marks
          : null;

    if (marks && marks.length) {
        const filled = marks.filter((mark) => mark && mark.filled);

        if (filled.length === 1) {
            const mark = filled[0];
            const byOption = Number.parseInt(
                mark.option ?? mark.index ?? mark.number,
                10
            );

            if (byOption >= 1 && byOption <= logisticas.length) {
                return logisticas[byOption - 1];
            }

            return (
                normalizeLogistica(mark.name || mark.selected_logistica) ||
                null
            );
        }
    }

    const optionNumber = Number.parseInt(
        option ?? source?.logistica_option ?? source?.option ?? "",
        10
    );

    if (optionNumber >= 1 && optionNumber <= logisticas.length) {
        return logisticas[optionNumber - 1];
    }

    if (typeof rawOrText === "string") {
        return normalizeLogistica(rawOrText);
    }

    return (
        normalizeLogistica(source?.selected_logistica) ||
        normalizeLogistica(source?.logistica) ||
        null
    );
}

function cleanFolio(text) {
    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    const digits = cleaned.replace(/[^\d]/g, "");
    return digits || cleaned;
}

function cleanSitio(text) {
    const cleaned = cleanOcrText(text).toUpperCase();

    if (!cleaned) {
        return null;
    }

    // Prefer MXCD-03 / MXCD03 style codes.
    const match = cleaned.match(
        /\b(MX[A-Z]{0,3}[- ]?\d{1,3})\b/
    );

    if (match) {
        return match[1].replace(/\s+/g, "");
    }

    // Reject obvious garbage from misaligned crops.
    if (/PESO|FECHA|FOLIO|SITIO|LOGISTICA/i.test(cleaned)) {
        return null;
    }

    return cleaned;
}

function looksLikeSitio(text) {
    return Boolean(
        cleanOcrText(text).match(/\bMX[A-Z]{0,3}[- ]?\d{1,3}\b/i)
    );
}

function looksLikeFolio(text) {
    const digits = cleanFolio(text);
    return Boolean(digits && /^\d{3,6}$/.test(digits));
}

function looksLikeFecha(text) {
    const cleaned = cleanOcrText(text);
    return Boolean(
        cleaned &&
            (/(\d{1,2})[-\/](\d{1,2})[-\/](\d{2,4})/.test(
                cleaned
            ) ||
                /(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})/.test(
                    cleaned
                ))
    );
}

function normalizeUnidadMedida(text) {
    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    const upper = cleaned.toUpperCase();

    if (/GRANEL/.test(upper)) {
        return "A granel";
    }

    if (/PACA/.test(upper)) {
        return "Pacas";
    }

    if (/GAYL|GOYL|GAYLORD|GOYLOARD/.test(upper)) {
        return "Gaylord's";
    }

    if (/BARCIN/.test(upper)) {
        return "Barcinas";
    }

    if (/PIEZ/.test(upper)) {
        return "Piezas";
    }

    return cleaned;
}

function normalizeMaterialName(text, otroText = null) {
    let cleaned = cleanOcrText(text);

    if (!cleaned && !otroText) {
        return null;
    }

    cleaned = cleaned
        ? cleaned.replace(/^\d+\s*[.\-–]?\s*/u, "").trim()
        : "";

    const upper = cleaned.toUpperCase();
    const detail = cleanOcrText(otroText);

    if (/^OTRO\b/.test(upper) || detail) {
        let label = detail;

        if (!label && cleaned) {
            label = cleaned.replace(/^otro\b\s*[:\-]?\s*/i, "").trim();
        }

        if (!label || /^otro$/i.test(label)) {
            return "Otro";
        }

        return `Otro: ${label}`;
    }

    if (/^PLAYO/.test(upper)) {
        return "Playo";
    }

    if (/^CART[OÓ]N/.test(upper)) {
        return "Carton";
    }

    if (/^RSU/.test(upper)) {
        return "RSU";
    }

    if (/^TARIMA/.test(upper)) {
        return "Tarima";
    }

    if (/TUBO/.test(upper)) {
        return "Tubo de carton";
    }

    if (/ORG[AÁ]NIC/.test(upper)) {
        return "Organicos";
    }

    if (/CHATARRA/.test(upper)) {
        return "Chatarra";
    }

    return cleaned || null;
}

function normalizePlacas(text) {
    if (text == null) {
        return null;
    }

    const raw = String(text).trim();

    // Check before cleanOcrText — models often return s/n, S/N, SN, sin numero.
    if (
        /^(s\s*\/\s*n|s\s*-\s*n|s\.?\s*n\.?|sn|sin\s*n[uú]mero|s\s*n)$/i.test(
            raw
        ) ||
        /\bs\s*\/\s*n\b/i.test(raw) ||
        /\bsin\s*n[uú]mero\b/i.test(raw)
    ) {
        return "S/N";
    }

    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    if (
        /^(s\s*\/\s*n|s\s*-\s*n|s\.?\s*n\.?|sn|sin\s*n[uú]mero)$/i.test(
            cleaned
        )
    ) {
        return "S/N";
    }

    return cleaned.replace(/\s+/g, "").toUpperCase();
}

function normalizeOperadorId(text) {
    const cleaned = cleanOcrText(text);

    if (!cleaned) {
        return null;
    }

    return cleaned.replace(/[:]/g, "-").replace(/\s+/g, "").toUpperCase();
}

function normalizeLogistica(text) {
    const cleaned = cleanOcrText(text).toUpperCase();

    if (!cleaned) {
        return null;
    }

    for (const name of logisticas) {
        if (cleaned === name || cleaned.includes(name)) {
            return name;
        }
    }

    // Common OCR misspellings near the form labels.
    if (/ROIS|ROSSE|ROSET/.test(cleaned)) {
        return "ROSSET";
    }

    if (/TREE|TREESSE/.test(cleaned)) {
        return "TREESEVER";
    }

    if (/BIO/.test(cleaned)) {
        return "BIOAMBIENTALISTIK";
    }

    if (/JUAN/.test(cleaned)) {
        return "JUAN CARLOS";
    }

    if (/SERRANO/.test(cleaned)) {
        return "SERRANO";
    }

    if (/TAGA/.test(cleaned)) {
        return "TAGA";
    }

    if (/PLASTIC/.test(cleaned)) {
        return "PLASTIC";
    }

    return null;
}

function normalizeFirmaResult(raw) {
    let value = raw?.value;

    if (typeof value === "string") {
        value = cleanOcrText(value) || null;
    } else {
        value = null;
    }

    // Signature ink counts even when the name is unreadable.
    if (raw?.filled || raw?.has_ink || raw?.hasInk) {
        return {
            filled: true,
            value: value || "Unknown"
        };
    }

    if (value) {
        return {
            filled: true,
            value
        };
    }

    return {
        filled: false,
        value: null
    };
}

function firmaFromText(text, hasInk = false) {
    const value = cleanOcrText(text);

    if (value) {
        return {
            filled: true,
            value
        };
    }

    if (hasInk) {
        return {
            filled: true,
            value: "Unknown"
        };
    }

    return {
        filled: false,
        value: null
    };
}

// -----------------------------------------------------------------------------
// OpenCV helpers
// -----------------------------------------------------------------------------

function loadImage(imageBuffer) {
    const img = cv.imdecode(imageBuffer);

    if (img.empty) {
        throw new Error("Unable to decode input image");
    }

    const gray =
        img.channels === 1
            ? img
            : img.cvtColor(cv.COLOR_BGR2GRAY);

    return { color: img, gray };
}

function rotate90Clockwise(mat) {
    try {
        if (typeof cv.rotate === "function") {
            return cv.rotate(mat, 0); // ROTATE_90_CLOCKWISE
        }
    } catch {
        // fall through
    }

    return mat.transpose().flip(1);
}

function rotateToLandscape(mat) {
    if (mat.cols >= mat.rows) {
        return mat;
    }

    // 90° clockwise for portrait phone photos of landscape forms.
    return rotate90Clockwise(mat);
}

function rotateToPortrait(mat) {
    if (mat.rows >= mat.cols) {
        return mat;
    }

    // Thermal receipts are portrait; landscape photos are usually sideways.
    return rotate90Clockwise(mat);
}

function findHomography(templateGray, inputGray) {
    const orb = new cv.ORBDetector({
        maxFeatures: 8000
    });

    const kp1 = orb.detect(templateGray);
    const des1 = orb.compute(templateGray, kp1);
    const kp2 = orb.detect(inputGray);
    const des2 = orb.compute(inputGray, kp2);

    if (!des1 || !des2 || des1.rows === 0 || des2.rows === 0) {
        throw new Error("No features found");
    }

    const matches = cv.matchKnnBruteForceHamming(des1, des2, 2);
    const good = [];

    for (const pair of matches) {
        if (pair.length < 2) {
            continue;
        }

        const [m, n] = pair;

        if (m.distance < 0.75 * n.distance) {
            good.push(m);
        }
    }

    if (good.length < 25) {
        throw new Error(`Not enough feature matches: ${good.length}`);
    }

    const srcPoints = good.map((m) => kp1[m.queryIdx].pt);
    const dstPoints = good.map((m) => kp2[m.trainIdx].pt);

    const result = cv.findHomography(
        dstPoints,
        srcPoints,
        cv.RANSAC,
        5
    );

    if (!result || !result.homography) {
        throw new Error("Unable to calculate homography");
    }

    let inliers = good.length;

    if (result.mask) {
        try {
            const data = result.mask.getDataAsArray();
            inliers = 0;

            for (const row of data) {
                for (const value of row) {
                    if (value) {
                        inliers += 1;
                    }
                }
            }
        } catch {
            // keep estimate
        }
    }

    return {
        homography: result.homography,
        inliers,
        matches: good.length
    };
}

function upscaleMat(mat, scale = 3) {
    if (!mat || scale <= 1) {
        return mat;
    }

    return mat.resize(
        Math.max(1, Math.round(mat.rows * scale)),
        Math.max(1, Math.round(mat.cols * scale)),
        0,
        0,
        cv.INTER_CUBIC
    );
}

function prepareForMatching(color) {
    // Low-res phone photos match the template much better when upscaled.
    const minSide = Math.min(color.rows, color.cols);

    if (minSide >= 900) {
        return color;
    }

    const scale = Math.min(2.5, 900 / minSide);
    return upscaleMat(color, scale);
}

function toInkGray(colorMat) {
    // Blue ballpoint is bright in B but dark in R/G. Averaging R+G
    // makes blue and black ink both look dark for radio detection.
    if (!colorMat || colorMat.channels === 1) {
        return colorMat;
    }

    try {
        const channels = colorMat.splitChannels();
        // OpenCV BGR: 0=B, 1=G, 2=R
        const g = channels[1];
        const r = channels[2];
        return g.add(r).div(2);
    } catch {
        return colorMat.cvtColor(cv.COLOR_BGR2GRAY);
    }
}

function warpImage(img, H, width = TARGET_W, height = TARGET_H) {
    return img.warpPerspective(
        H,
        new cv.Size(width, height),
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Vec3(255, 255, 255)
    );
}

function matToJpeg(mat) {
    try {
        return cv.imencode(".jpg", mat, [1, 90]); // IMWRITE_JPEG_QUALITY = 1
    } catch {
        return cv.imencode(".jpg", mat);
    }
}

function matToPng(mat) {
    return cv.imencode(".png", mat);
}

function cropMat(img, x1, y1, x2, y2) {
    const left = Math.max(0, Math.min(x1, img.cols - 1));
    const top = Math.max(0, Math.min(y1, img.rows - 1));
    const right = Math.max(left + 1, Math.min(x2, img.cols));
    const bottom = Math.max(top + 1, Math.min(y2, img.rows));

    return img.getRegion(
        new cv.Rect(left, top, right - left, bottom - top)
    );
}

function clipLooksBlank(mat) {
    try {
        const gray =
            mat.channels === 1
                ? mat
                : mat.cvtColor(cv.COLOR_BGR2GRAY);
        const data = gray.getDataAsArray();
        let dark = 0;
        let total = 0;
        let sum = 0;

        for (const row of data) {
            for (const value of row) {
                total += 1;
                sum += value;

                if (value < 140) {
                    dark += 1;
                }
            }
        }

        if (!total) {
            return true;
        }

        // Mostly white paper with almost no ink strokes.
        return sum / total > 200 && dark / total < 0.02;
    } catch {
        return false;
    }
}

async function ocrClip(mat, fieldName) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    if (clipLooksBlank(mat)) {
        return "";
    }

    // Tiny crops OCR poorly; enlarge before Vision.
    const enlarged = upscaleMat(mat, 4);
    const buffer = matToPng(enlarged);
    const base64 = buffer.toString("base64");
    const hint =
        FIELD_HINTS[fieldName] ||
        "Extract only the text in this box.";

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 40,
        messages: [
            {
                role: "system",
                content:
                    "You are an OCR engine. Return only the extracted text. " +
                    "No labels, no quotes, no explanation, no apologies. " +
                    "If the box is blank or unreadable, return an empty string and nothing else."
            },
            {
                role: "user",
                content: [
                    { type: "text", text: hint },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/png;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    return cleanOcrText(
        response.choices?.[0]?.message?.content || ""
    );
}

async function ocrFieldClips(alignedColor) {
    const entries = await Promise.all(
        Object.entries(fieldClips).map(
            async ([name, [x1, y1, x2, y2]]) => {
                const clip = cropMat(
                    alignedColor,
                    x1,
                    y1,
                    x2,
                    y2
                );
                const blank = clipLooksBlank(clip);
                const text = blank ? "" : await ocrClip(clip, name);
                return [name, { text, hasInk: !blank }];
            }
        )
    );

    return Object.fromEntries(entries);
}

function sampleGray(gray, x, y) {
    if (
        x < 0 ||
        y < 0 ||
        x >= gray.cols ||
        y >= gray.rows
    ) {
        return null;
    }

    return gray.at(y, x);
}

// Filled radio = dark center + lighter ring. Avoids nearby text (e.g. PLASTIC).
function filledCircleScore(gray, x, y) {
    let coreSum = 0;
    let coreCount = 0;
    let ringSum = 0;
    let ringCount = 0;

    for (let dy = -8; dy <= 8; dy++) {
        for (let dx = -8; dx <= 8; dx++) {
            const dist2 = dx * dx + dy * dy;
            const value = sampleGray(gray, x + dx, y + dy);

            if (value == null) {
                continue;
            }

            if (dist2 <= 9) {
                // r <= 3
                coreSum += value;
                coreCount += 1;
            } else if (dist2 >= 25 && dist2 <= 64) {
                // 5 <= r <= 8
                ringSum += value;
                ringCount += 1;
            }
        }
    }

    if (!coreCount || !ringCount) {
        return null;
    }

    const core = coreSum / coreCount;
    const ring = ringSum / ringCount;
    const contrast = ring - core;

    // Blurry phone photos often darken the ring (~100-130), so do not
    // require a bright white ring. Require a dark core + positive contrast.
    if (core > 45 || contrast < 50) {
        return null;
    }

    return {
        core,
        ring,
        contrast
    };
}

function bestFilledNear(gray, x, y, radius = 8) {
    let best = null;

    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            const score = filledCircleScore(gray, x + dx, y + dy);

            if (!score) {
                continue;
            }

            if (!best || score.core < best.core) {
                best = {
                    ...score,
                    x: x + dx,
                    y: y + dy
                };
            }
        }
    }

    return best;
}

function detectSelectedOption(gray, positions, names) {
    const scored = [];

    for (let i = 0; i < positions.length; i++) {
        const [x, y] = positions[i];
        const score = bestFilledNear(gray, x, y, 8);

        if (!score) {
            continue;
        }

        scored.push({
            name: names[i],
            ...score
        });
    }

    if (!scored.length) {
        return null;
    }

    scored.sort((a, b) => a.core - b.core);

    const best = scored[0];
    const second = scored[1];

    // Require a clear winner so blue smudges / warp noise do not flip LOGÍSTICA.
    if (second && best.core > 18 && second.core - best.core < 12) {
        return null;
    }

    return best.name;
}

function alignDocument(
    inputBuffer,
    {
        templatePath = TEMPLATE_PATH,
        width = TARGET_W,
        height = TARGET_H,
        orientation = "landscape"
    } = {}
) {
    const templateBuffer = fs.readFileSync(templatePath);
    const template = loadImage(templateBuffer);
    let { color } = loadImage(inputBuffer);

    color =
        orientation === "portrait"
            ? rotateToPortrait(color)
            : rotateToLandscape(color);

    color = prepareForMatching(color);

    const gray =
        color.channels === 1
            ? color
            : color.cvtColor(cv.COLOR_BGR2GRAY);

    const { homography: H, inliers, matches } = findHomography(
        template.gray,
        gray
    );

    if (inliers < MIN_ALIGNMENT_INLIERS) {
        throw new Error(
            `Weak alignment: ${inliers} inliers from ${matches} matches`
        );
    }

    const alignedColor = warpImage(color, H, width, height);
    const alignedGray =
        alignedColor.channels === 1
            ? alignedColor
            : alignedColor.cvtColor(cv.COLOR_BGR2GRAY);

    return {
        color: alignedColor,
        gray: alignedGray,
        inkGray: toInkGray(alignedColor),
        inliers,
        aligned: true
    };
}

// -----------------------------------------------------------------------------
// Full-page OpenAI Vision extraction
// -----------------------------------------------------------------------------

async function extractDocumentWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 1600,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You extract data from Plastic Trade ORDEN DE SALIDA forms. " +
                    "Return JSON only with this exact shape:\n" +
                    "{\n" +
                    '  "documento": {"sitio": string|null, "folio": string|null, "fecha": "YYYY-MM-DD"|null},\n' +
                    '  "selected_logistica": string|null,\n' +
                    '  "logistica_option": number|null,\n' +
                    '  "materials": [{"material": string, "otro_text": string|null, "cantidad": number|null, "unidad": string|null}],\n' +
                    '  "selected_unidad": string|null,\n' +
                    '  "operador": {\n' +
                    '    "id_operador": string|null,\n' +
                    '    "nombre": string|null,\n' +
                    '    "placas_vehiculo": string|null,\n' +
                    '    "placas_caja_remolque": string|null,\n' +
                    '    "numero_marchamo": string|null\n' +
                    "  },\n" +
                    '  "horarios": {"hora_entrada": string|null, "hora_salida": string|null},\n' +
                    '  "firmas": {\n' +
                    '    "elaboro": {"filled": boolean, "value": string|null},\n' +
                    '    "supervisor": {"filled": boolean, "value": string|null},\n' +
                    '    "autorizo": {"filled": boolean, "value": string|null},\n' +
                    '    "operador": {"filled": boolean, "value": string|null}\n' +
                    "  }\n" +
                    "}\n" +
                    "Rules:\n" +
                    "- Only include materials whose LEFT material checkbox is selected AND that have a handwritten quantity.\n" +
                    "- Read EACH material row independently. Do not copy UNIDAD DE MEDIDA from another row.\n" +
                    "- For each selected row, unidad is ONLY the filled UNIDAD DE MEDIDA circle in THAT same row " +
                    "(A granel, Pacas, Gaylord's, Barcinas, Piezas). " +
                    "For Carton options are usually A granel then Gaylord's — choose only the ink-filled circle.\n" +
                    "- Also include udm_marks: [{name, filled}] for every UDM option printed in that row.\n" +
                    "- Standard material names in \"material\": Playo, Carton, RSU, Tarima, Tubo de carton, Organicos, Chatarra, Otro.\n" +
                    "- For Otro: set material=\"Otro\" and put the handwritten label in otro_text (e.g. Goyloards). " +
                    "otro_text is NOT the unidad.\n" +
                    "- Never copy unidad from another row.\n" +
                    "- Include every selected material row (Playo, Carton, Otro, etc.).\n" +
                    "- selected_logistica: look ONLY at which LOGÍSTICA radio circle is filled (black or blue ink dot). " +
                    "Each circle sits immediately LEFT of its name. The fill belongs to the name on its RIGHT, not the previous name. " +
                    "A dot between TAGA and SERRANO is SERRANO (2), not TAGA. " +
                    "A dot between ROISROSE/ROSSET and PLASTIC is PLASTIC (6). " +
                    "A dot between TREESEVER and ROISROSE is ROSSET (5). " +
                    "A dot between JUAN CARLOS and TREESEVER is TREESEVER (4).\n" +
                    "Also return logistica_option as 1-7. " +
                    "1 TAGA, 2 SERRANO, 3 JUAN CARLOS, 4 TREESEVER, 5 ROSSET, 6 PLASTIC, 7 BIOAMBIENTALISTIK. " +
                    "Inspect every circle carefully, including 7 BIOAMBIENTALISTIK on the far right. " +
                    "Empty outline is NOT selected. Do not default to TAGA; TAGA only if the leftmost circle is filled.\n" +
                    "- documento.sitio: value under SITIO (e.g. MXCD-03). Keep hyphen if present.\n" +
                    "- documento.folio: digits only from FOLIO (N° 0251 -> 0251). May be red ink.\n" +
                    "- documento.fecha: handwritten date under FECHA as YYYY-MM-DD.\n" +
                    "- selected_unidad: look ONLY at which UNIDAD DE TRANSPORTE radio circle is filled (black or blue). " +
                    "Options in order: Caja seca, Tolva 30m3, Remolque, Torthon, Cartucho, Olla 17m3, Camioneta, Tolva 7m3, Contenedores CGR. " +
                    "Return that exact selected name. Do not guess from nearby text.\n" +
                    "- folio: prefer digits only (N° 0001 -> 0001).\n" +
                    "- horarios: these boxes often contain ONLY a time like 17:00 hrs. " +
                    "Return the time as HH:mm (or YYYY-MM-DD HH:mm if a real date is written). " +
                    "Do NOT invent dates. Empty box => null.\n" +
                    "- operador fields: empty box => null. " +
                    "IMPORTANT: if Placas de caja/remolque (or vehicle plates) show S/N, s/n, SN, or Sin Número, " +
                    "return exactly \"S/N\" — that is a real value, NOT empty/null. " +
                    "Read operator names carefully (Colina vs Cline). Never apologize or explain.\n" +
                    "- firmas: filled=true if ANY signature ink/scribble is present in the box, even if the name is unreadable; " +
                    "value=readable name or Unknown; empty box with no ink => filled=false, value=null.\n" +
                    "- Read handwriting carefully (9 vs Y, 6 vs 0, 5 vs S, Colina vs Cline). Do not invent values. Never return apology text."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Extract all filled fields from this ORDEN DE SALIDA image."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        throw new Error("OpenAI returned invalid JSON for document extraction");
    }
}

function pickFirma(fields, firmasRaw, inkMap, cropKey, ...visionKeys) {
    if (Object.prototype.hasOwnProperty.call(fields || {}, cropKey)) {
        return firmaFromText(
            fields[cropKey],
            Boolean(inkMap?.[cropKey])
        );
    }

    for (const key of visionKeys) {
        if (firmasRaw?.[key] != null) {
            return normalizeFirmaResult(firmasRaw[key]);
        }
    }

    return {
        filled: false,
        value: null
    };
}

function unwrapFieldClips(fieldOverride) {
    if (!fieldOverride) {
        return { fields: {}, ink: {} };
    }

    const fields = {};
    const ink = {};

    for (const [key, value] of Object.entries(fieldOverride)) {
        if (value && typeof value === "object" && "text" in value) {
            fields[key] = value.text;
            ink[key] = Boolean(value.hasInk);
        } else {
            fields[key] = value;
        }
    }

    return { fields, ink };
}

async function extractHeaderLogisticaWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 300,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You read the top-right header and LOGÍSTICA row of a Plastic Trade ORDEN DE SALIDA. " +
                    "Return JSON only:\n" +
                    "{\n" +
                    '  "sitio": string|null,\n' +
                    '  "folio": string|null,\n' +
                    '  "fecha": "YYYY-MM-DD"|null,\n' +
                    '  "logistica_option": number|null,\n' +
                    '  "selected_logistica": string|null\n' +
                    "}\n" +
                    "Rules:\n" +
                    "- sitio is under SITIO (e.g. MXCD-03). Keep letters, digits, and hyphen.\n" +
                    "- folio is under FOLIO; digits only (N° 0251 -> 0251). May be red.\n" +
                    "- fecha is the handwritten date under FECHA.\n" +
                    "- LOGÍSTICA options numbered 1..7 left-to-right: " +
                    "1 TAGA, 2 SERRANO, 3 JUAN CARLOS, 4 TREESEVER, 5 ROSSET, 6 PLASTIC, 7 BIOAMBIENTALISTIK.\n" +
                    "- Each filled circle belongs to the name immediately to its RIGHT. " +
                    "The circle between TAGA and SERRANO is SERRANO (option 2), not TAGA.\n" +
                    "- logistica_option = the NUMBER of the circle filled with ink (black or blue dot). " +
                    "Inspect every circle carefully, including option 7 BIOAMBIENTALISTIK on the far right. " +
                    "Empty ring != selected. Only one is selected.\n" +
                    "- Do not default to TAGA. TAGA is selected only if the leftmost circle itself is filled.\n" +
                    "- selected_logistica = exact name for that number.\n" +
                    "- Do not invent values. Never apologize."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Extract SITIO, FOLIO, FECHA, and which LOGÍSTICA option number is filled."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        return {};
    }
}

async function extractLogisticaOnlyWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 400,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "This crop shows the LOGÍSTICA radio row from a Plastic Trade ORDEN DE SALIDA. " +
                    "Return JSON only:\n" +
                    "{\n" +
                    '  "logistica_marks": [\n' +
                    '    {"option":1,"name":"TAGA","filled":boolean},\n' +
                    '    {"option":2,"name":"SERRANO","filled":boolean},\n' +
                    '    {"option":3,"name":"JUAN CARLOS","filled":boolean},\n' +
                    '    {"option":4,"name":"TREESEVER","filled":boolean},\n' +
                    '    {"option":5,"name":"ROSSET","filled":boolean},\n' +
                    '    {"option":6,"name":"PLASTIC","filled":boolean},\n' +
                    '    {"option":7,"name":"BIOAMBIENTALISTIK","filled":boolean}\n' +
                    "  ],\n" +
                    '  "logistica_option": number|null,\n' +
                    '  "selected_logistica": string|null\n' +
                    "}\n" +
                    "Rules:\n" +
                    "- Each option is printed as: (circle) number.- NAME. " +
                    "The circle belongs to the NAME immediately to its RIGHT, never the previous name.\n" +
                    "- Circle 1 is left of TAGA. Circle 2 is left of SERRANO (between TAGA and SERRANO). " +
                    "Circle 3 left of JUAN CARLOS. Circle 4 left of TREESEVER. Circle 5 left of ROSSET. " +
                    "Circle 6 left of PLASTIC. Circle 7 left of BIOAMBIENTALISTIK.\n" +
                    "- filled=true ONLY if that option's own circle has solid blue/black ink inside.\n" +
                    "- Empty ring / white center => filled=false.\n" +
                    "- Exactly ONE option should be filled.\n" +
                    "- Do not default to TAGA. Do not assign a fill to the previous label.\n" +
                    "- logistica_option = that option number. selected_logistica = that name.\n" +
                    "- Do not guess from company logos, wood background, FECHA handwriting, or the words PLASTIC TRADE. " +
                    "Only the filled LOGÍSTICA circle counts."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text:
                            "Mark filled/empty for all 7 LOGÍSTICA circles, then return the selected option number."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        return {};
    }
}

async function extractMaterialsWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 900,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "Extract selected material rows from an ORDEN DE SALIDA materials table. " +
                    "Return JSON only:\n" +
                    '{"materials":[{"material":string,"otro_text":string|null,"cantidad":number|null,' +
                    '"udm_marks":[{"name":string,"filled":boolean}],"unidad":string|null}]}\n' +
                    "Rules:\n" +
                    "- Include a row only if its LEFT material circle is filled AND it has a handwritten quantity.\n" +
                    "- material is one of: Playo, Carton, RSU, Tarima, Tubo de carton, Organicos, Chatarra, Otro.\n" +
                    "- For Otro: material=\"Otro\", otro_text=handwritten label next to Otro (e.g. Goyloards).\n" +
                    "- For EACH selected row, list EVERY UNIDAD DE MEDIDA option printed in that same row in udm_marks " +
                    "(left-to-right). Set filled=true ONLY if that option's circle has blue/black ink. Empty ring => filled=false.\n" +
                    "- Common row layouts:\n" +
                    "  Playo: A granel, Pacas, Gaylord's, Barcinas\n" +
                    "  Carton: A granel, Gaylord's\n" +
                    "  RSU: A granel\n" +
                    "  Tarima: Piezas\n" +
                    "  Otro: A granel, Piezas, Gaylord's\n" +
                    "- unidad MUST be the name of the ONE udm_marks item with filled=true in that row.\n" +
                    "- Critical examples:\n" +
                    "  Carton: if A granel filled and Gaylord's empty => unidad=A granel.\n" +
                    "  Playo: if Gaylord's filled and A granel/Pacas/Barcinas empty => unidad=Gaylord's.\n" +
                    "  Playo layout left-to-right: A granel, Pacas, Gaylord's, Barcinas.\n" +
                    "- Do NOT mark multiple UDM options filled in one row. Empty ring => filled=false.\n" +
                    "- Ignore the vertical UNIDAD DE TRANSPORTE list on the right (Caja seca, Tolva, Remolque, etc).\n" +
                    "- Never copy unidad from another row. Never put otro_text into unidad."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text:
                            "For each selected material row, mark every UDM circle filled/empty, " +
                            "then set unidad to the filled one only."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        return { materials: [] };
    }
}

function resolveMaterialUnidad(item) {
    const marks = Array.isArray(item?.udm_marks)
        ? item.udm_marks
        : Array.isArray(item?.udmMarks)
          ? item.udmMarks
          : null;

    const fromUnidad = normalizeUnidadMedida(item?.unidad);

    if (marks && marks.length) {
        const filled = marks
            .filter((mark) => mark && mark.filled)
            .map((mark) => normalizeUnidadMedida(mark.name))
            .filter(Boolean);

        if (filled.length === 1) {
            return filled[0];
        }

        if (filled.length > 1) {
            // If model over-marked, trust unidad when it matches one filled mark.
            if (fromUnidad && filled.includes(fromUnidad)) {
                return fromUnidad;
            }

            // Otherwise keep the last filled mark (rightmost intentional mark).
            return filled[filled.length - 1];
        }
    }

    return fromUnidad;
}

async function extractOperadorWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 300,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "Extract operator / vehicle fields from the bottom of an ORDEN DE SALIDA. " +
                    "Return JSON only:\n" +
                    "{\n" +
                    '  "hora_entrada": string|null,\n' +
                    '  "hora_salida": string|null,\n' +
                    '  "id_operador": string|null,\n' +
                    '  "nombre": string|null,\n' +
                    '  "placas_vehiculo": string|null,\n' +
                    '  "placas_caja_remolque": string|null,\n' +
                    '  "numero_marchamo": string|null\n' +
                    "}\n" +
                    "Rules:\n" +
                    "- If Placas de caja/remolque is written as S/N, s/n, SN, or Sin Número, return \"S/N\".\n" +
                    "- S/N is a valid filled value — never convert it to null.\n" +
                    "- Times may be HH:mm only (e.g. 16:00).\n" +
                    "- id_operador is usually PT-XX-NNN. Read XX carefully: " +
                    "TG, SR, JC, TH, RH, PT, or BIO. Do not confuse SR with TG or TH, or RH with TH.\n" +
                    "- Empty box => null. Never apologize."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Extract times, operator ID/name, plates, and marchamo. Keep S/N if written."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        return {};
    }
}

function pickBestDocumentoField(cropValue, headerValue, visionValue, kind) {
    const candidates = [headerValue, cropValue, visionValue]
        .map((value) => cleanOcrText(value))
        .filter(Boolean);

    const checker =
        kind === "sitio"
            ? looksLikeSitio
            : kind === "folio"
              ? looksLikeFolio
              : looksLikeFecha;

    for (const value of candidates) {
        if (checker(value)) {
            if (kind === "sitio") {
                return cleanSitio(value);
            }

            if (kind === "folio") {
                return cleanFolio(value);
            }

            return parseDate(value);
        }
    }

    if (kind === "sitio") {
        return cleanSitio(headerValue || cropValue || visionValue);
    }

    if (kind === "folio") {
        return cleanFolio(headerValue || cropValue || visionValue);
    }

    return parseDate(headerValue || cropValue || visionValue);
}

function buildResponse(raw, overrides = {}) {
    const unwrapped = unwrapFieldClips(overrides.fields);
    const fields = unwrapped.fields;
    const inkMap = unwrapped.ink;
    const header = overrides.header || {};
    const firmasRaw = raw?.firmas || {};

    const materials = Array.isArray(raw?.materials)
        ? raw.materials
            .map((item) => {
                const cantidad = parseInteger(item?.cantidad);
                const material = normalizeMaterialName(
                    item?.material,
                    item?.otro_text || item?.otroText
                );
                const unidad = resolveMaterialUnidad(item);

                return {
                    material,
                    cantidad,
                    unidad
                };
            })
            .filter(
                (item) =>
                    item.material &&
                    item.cantidad != null &&
                    item.cantidad > 0
            )
        : [];

    const selectedLogistica =
        resolveLogistica(overrides.logistica) ||
        resolveLogistica(header) ||
        resolveLogistica(raw) ||
        null;

    const documento = {
        sitio: pickBestDocumentoField(
            fields.sitio,
            header.sitio,
            raw?.documento?.sitio,
            "sitio"
        ),
        folio: pickBestDocumentoField(
            fields.folio,
            header.folio,
            raw?.documento?.folio,
            "folio"
        ),
        fecha: pickBestDocumentoField(
            fields.fecha,
            header.fecha,
            raw?.documento?.fecha,
            "fecha"
        )
    };

    // Strict whitelist — never pass through raw Vision / legacy keys.
    return {
        documento,
        selected_logistica: selectedLogistica,
        materials,
        selected_unidad:
            overrides.unidad ||
            cleanOcrText(raw?.selected_unidad) ||
            null,
        operador: {
            id_operador:
                normalizeOperadorId(fields.id_operador) ||
                normalizeOperadorId(raw?.operador?.id_operador) ||
                null,
            nombre:
                cleanOcrText(fields.nombre) ||
                cleanOcrText(raw?.operador?.nombre) ||
                null,
            placas_vehiculo:
                normalizePlacas(fields.placas_vehiculo) ||
                normalizePlacas(raw?.operador?.placas_vehiculo) ||
                null,
            placas_caja_remolque:
                normalizePlacas(fields.placas_caja_remolque) ||
                normalizePlacas(
                    raw?.operador?.placas_caja_remolque
                ) ||
                null,
            numero_marchamo:
                cleanOcrText(fields.numero_marchamo) ||
                cleanOcrText(raw?.operador?.numero_marchamo) ||
                null
        },
        horarios: {
            hora_entrada: parseHorario(
                fields.hora_entrada ||
                    raw?.horarios?.hora_entrada ||
                    raw?.cliente_de_servicio
                        ?.fecha_hora_entrada_sitio,
                documento.fecha
            ),
            hora_salida: parseHorario(
                fields.hora_salida ||
                    raw?.horarios?.hora_salida ||
                    raw?.cliente_de_servicio
                        ?.fecha_hora_salida_sitio,
                documento.fecha
            )
        },
        firmas: {
            elaboro: pickFirma(
                fields,
                firmasRaw,
                inkMap,
                "elaboro",
                "elaboro",
                "elaboro_plastict"
            ),
            supervisor: pickFirma(
                fields,
                firmasRaw,
                inkMap,
                "supervisor",
                "supervisor",
                "responsable_supervisor"
            ),
            autorizo: pickFirma(
                fields,
                firmasRaw,
                inkMap,
                "autorizo",
                "autorizo",
                "autoriza_melii"
            ),
            operador: pickFirma(
                fields,
                firmasRaw,
                inkMap,
                "operador_firma",
                "operador",
                "recibio_y_entrego_operador"
            )
        }
    };
}

// -----------------------------------------------------------------------------
// Main document processing
// -----------------------------------------------------------------------------

async function prepareImageWithSharp(inputBuffer, orientation = "landscape") {
    let image = sharp(inputBuffer).rotate(); // honor EXIF
    const meta = await image.metadata();
    let width = meta.width || 0;
    let height = meta.height || 0;

    if (orientation === "landscape" && height > width) {
        image = image.rotate(90);
        [width, height] = [height, width];
    } else if (orientation === "portrait" && width > height) {
        image = image.rotate(90);
        [width, height] = [height, width];
    }

    const jpegBuffer = await image
        .jpeg({ quality: 90 })
        .toBuffer();

    return { jpegBuffer, width, height };
}

async function findMaterialsHeaderBarY(jpegBuffer, width, height) {
    // Locate the black "Material / Cantidad / Unidad de medida" bar.
    const tw = 420;
    const th = Math.max(1, Math.round((height * tw) / Math.max(width, 1)));
    const { data, info } = await sharp(jpegBuffer)
        .resize(tw, th)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const channels = info.channels;
    const scores = new Array(th).fill(0);
    const x0 = Math.floor(tw * 0.08);
    const x1 = Math.floor(tw * 0.78);

    for (let y = 0; y < th; y++) {
        let dark = 0;
        for (let x = x0; x < x1; x++) {
            const idx = (y * tw + x) * channels;
            const avg = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
            if (avg < 55) {
                dark += 1;
            }
        }
        scores[y] = dark;
    }

    let bestY = Math.floor(th * 0.28);
    let best = 0;

    for (let y = Math.floor(th * 0.14); y < Math.floor(th * 0.58); y++) {
        const s =
            scores[y] +
            (scores[y - 1] || 0) +
            (scores[y + 1] || 0);
        if (s > best) {
            best = s;
            bestY = y;
        }
    }

    return {
        yFrac: bestY / th,
        score: best
    };
}

async function processDocumentVisionOnly(inputBuffer) {
    console.log("Processing sheet in Vision-only mode (no OpenCV)");

    const prepared = await prepareImageWithSharp(
        inputBuffer,
        "landscape"
    );

    const headerBar = await findMaterialsHeaderBarY(
        prepared.jpegBuffer,
        prepared.width,
        prepared.height
    );
    const barY = Math.floor(headerBar.yFrac * prepared.height);
    console.log("Materials header bar:", headerBar);

    const topHeight = Math.max(1, Math.floor(prepared.height * 0.4));

    // LOGÍSTICA sits just above the materials header bar.
    const logHeight = Math.max(1, Math.floor(prepared.height * 0.075));
    const logTop = Math.max(0, barY - logHeight);
    const logWidth = Math.max(1, Math.floor(prepared.width * 0.96));

    const headerJpeg = await sharp(prepared.jpegBuffer)
        .extract({
            left: 0,
            top: 0,
            width: prepared.width,
            height: topHeight
        })
        .resize({ width: Math.min(1600, prepared.width * 2) })
        .jpeg({ quality: 92 })
        .toBuffer();

    const logisticaJpeg = await sharp(prepared.jpegBuffer)
        .extract({
            left: 0,
            top: logTop,
            width: logWidth,
            height: Math.min(
                logHeight + Math.floor(prepared.height * 0.015),
                prepared.height - logTop
            )
        })
        .resize({ width: Math.min(2200, logWidth * 3) })
        .jpeg({ quality: 95 })
        .toBuffer();

    const materialsTop = Math.max(0, barY - Math.floor(prepared.height * 0.01));
    const materialsHeight = Math.max(
        1,
        Math.floor(prepared.height * 0.45)
    );

    const materialsJpeg = await sharp(prepared.jpegBuffer)
        .extract({
            left: 0,
            top: materialsTop,
            width: prepared.width,
            height: Math.min(
                materialsHeight,
                prepared.height - materialsTop
            )
        })
        .resize({ width: Math.min(1800, prepared.width * 2) })
        .jpeg({ quality: 95 })
        .toBuffer();

    const operadorTop = Math.max(
        0,
        Math.floor(prepared.height * 0.62)
    );
    const operadorJpeg = await sharp(prepared.jpegBuffer)
        .extract({
            left: 0,
            top: operadorTop,
            width: prepared.width,
            height: prepared.height - operadorTop
        })
        .resize({ width: Math.min(1800, prepared.width * 2) })
        .jpeg({ quality: 95 })
        .toBuffer();

    const [
        headerOverride,
        logisticaOnly,
        materialsOnly,
        operadorOnly,
        extracted
    ] = await Promise.all([
        extractHeaderLogisticaWithVision(headerJpeg),
        extractLogisticaOnlyWithVision(logisticaJpeg),
        extractMaterialsWithVision(materialsJpeg),
        extractOperadorWithVision(operadorJpeg),
        extractDocumentWithVision(prepared.jpegBuffer)
    ]);

    const logisticaHint = logisticaFromOperadorId(
        operadorOnly?.id_operador || extracted?.operador?.id_operador
    );
    const logisticaInk = await detectLogisticaByBlueInk(logisticaJpeg);
    console.log("LOGÍSTICA blue-ink:", logisticaInk);
    console.log("LOGÍSTICA from operator ID:", logisticaHint);

    const header = {
        ...headerOverride,
        logistica_marks:
            logisticaOnly?.logistica_marks ||
            logisticaOnly?.marks ||
            headerOverride?.logistica_marks ||
            null,
        logistica_option:
            logisticaInk?.logistica_option ??
            logisticaOnly?.logistica_option ??
            headerOverride?.logistica_option ??
            extracted?.logistica_option ??
            null,
        selected_logistica:
            pickSelectedLogistica(
                logisticaInk,
                logisticaOnly,
                logisticaHint
            ) ||
            pickSelectedLogistica(
                logisticaInk,
                headerOverride,
                logisticaHint
            ) ||
            pickSelectedLogistica(
                logisticaInk,
                extracted,
                logisticaHint
            ) ||
            logisticaHint ||
            null
    };

    const logisticaFromMarks = header.selected_logistica;

    if (
        Array.isArray(materialsOnly?.materials) &&
        materialsOnly.materials.length
    ) {
        extracted.materials = await refineMaterialsUnidadByBlueInk(
            materialsOnly.materials,
            materialsJpeg
        );
    }

    if (operadorOnly && typeof operadorOnly === "object") {
        extracted.operador = {
            ...(extracted.operador || {}),
            id_operador:
                operadorOnly.id_operador ||
                extracted.operador?.id_operador ||
                null,
            nombre:
                operadorOnly.nombre ||
                extracted.operador?.nombre ||
                null,
            placas_vehiculo:
                operadorOnly.placas_vehiculo ||
                extracted.operador?.placas_vehiculo ||
                null,
            placas_caja_remolque:
                operadorOnly.placas_caja_remolque ||
                extracted.operador?.placas_caja_remolque ||
                null,
            numero_marchamo:
                operadorOnly.numero_marchamo ||
                extracted.operador?.numero_marchamo ||
                null
        };

        extracted.horarios = {
            ...(extracted.horarios || {}),
            hora_entrada:
                operadorOnly.hora_entrada ||
                extracted.horarios?.hora_entrada ||
                null,
            hora_salida:
                operadorOnly.hora_salida ||
                extracted.horarios?.hora_salida ||
                null
        };
    }

    console.log("Header Vision:", headerOverride);
    console.log("LOGÍSTICA-only Vision:", logisticaOnly);
    console.log("Materials-only Vision:", materialsOnly);
    console.log("Operador-only Vision:", operadorOnly);

    return buildResponse(extracted, {
        header,
        logistica: logisticaFromMarks
    });
}

async function processKilogramVisionOnly(inputBuffer) {
    console.log("Processing kilogram in Vision-only mode (no OpenCV)");

    const prepared = await prepareImageWithSharp(
        inputBuffer,
        "portrait"
    );
    const extracted = await extractKilogramWithVision(
        prepared.jpegBuffer
    );

    return buildKilogramResponse(extracted);
}

async function processDocument(inputBuffer) {
    if (!cv) {
        return processDocumentVisionOnly(inputBuffer);
    }

    if (!fs.existsSync(TEMPLATE_PATH)) {
        throw new Error(
            `Template file not found: ${TEMPLATE_PATH}`
        );
    }

    let imageForOcr;
    let alignedGray = null;
    let alignedInkGray = null;
    let alignedColor = null;
    let alignmentStrong = false;

    try {
        const aligned = alignDocument(inputBuffer);
        imageForOcr = aligned.color;
        alignedColor = aligned.color;
        alignedGray = aligned.gray;
        alignedInkGray = aligned.inkGray || aligned.gray;
        alignmentStrong = true;
        console.log(
            "Document aligned to template, inliers:",
            aligned.inliers
        );
    } catch (error) {
        console.warn(
            "Alignment failed, using orientation-normalized original:",
            error.message
        );

        let { color } = loadImage(inputBuffer);
        color = rotateToLandscape(color);
        imageForOcr = color;
    }

    // Prefer OpenCV filled-circle detection only when warp is trustworthy.
    // Use ink-gray so blue ballpoint fills count as dark.
    let logisticaOverride = null;
    let unidadOverride = null;

    if (alignmentStrong && alignedInkGray) {
        logisticaOverride = detectSelectedOption(
            alignedInkGray,
            logisticas_cp,
            logisticas
        );

        if (logisticaOverride) {
            console.log(
                "LOGÍSTICA detected by checkbox:",
                logisticaOverride
            );
        }

        unidadOverride = detectSelectedOption(
            alignedInkGray,
            unidads_cp,
            unidads
        );

        if (unidadOverride) {
            console.log(
                "UNIDAD DE TRANSPORTE detected by checkbox:",
                unidadOverride
            );
        }
    }

    // Focused Vision on SITIO / FOLIO / FECHA / LOGÍSTICA (most failure-prone).
    let headerOverride = {};

    try {
        let headerMat = imageForOcr;
        let logisticaMat = null;

        if (alignedColor) {
            // Top band covering SITIO/FOLIO/FECHA and LOGÍSTICA.
            headerMat = cropMat(alignedColor, 20, 90, 1110, 260);
            logisticaMat = cropMat(alignedColor, 20, 210, 1100, 255);
        } else {
            // Top half of the unaligned landscape photo.
            const top = Math.max(1, Math.floor(imageForOcr.rows * 0.45));
            headerMat = cropMat(
                imageForOcr,
                0,
                0,
                imageForOcr.cols,
                top
            );
        }

        headerMat = upscaleMat(headerMat, 2);
        const headerRaw = await extractHeaderLogisticaWithVision(
            matToJpeg(headerMat)
        );

        let logisticaOnly = {};

        if (logisticaMat) {
            logisticaOnly = await extractLogisticaOnlyWithVision(
                matToJpeg(upscaleMat(logisticaMat, 3))
            );
        }

        headerOverride = {
            ...headerRaw,
            logistica_option:
                logisticaOnly?.logistica_option ??
                headerRaw?.logistica_option ??
                null,
            selected_logistica:
                logisticaOnly?.selected_logistica ||
                headerRaw?.selected_logistica ||
                null
        };

        console.log("Header/LOGÍSTICA Vision:", headerOverride);
    } catch (error) {
        console.warn(
            "Header/LOGÍSTICA Vision failed:",
            error.message
        );
    }

    // Crop OCR for fixed field boxes on the aligned template.
    let fieldOverride = null;

    if (alignedColor) {
        fieldOverride = await ocrFieldClips(alignedColor);
        console.log("Field crop OCR:", fieldOverride);
    }

    const jpegBuffer = matToJpeg(imageForOcr);
    const extracted = await extractDocumentWithVision(jpegBuffer);

    // Prefer OpenCV checkbox when available; otherwise Vision option number.
    const logisticaFinal =
        logisticaOverride ||
        resolveLogistica(headerOverride) ||
        resolveLogistica(extracted) ||
        null;

    return buildResponse(extracted, {
        logistica: logisticaFinal,
        unidad: unidadOverride,
        fields: fieldOverride,
        header: headerOverride
    });
}

// -----------------------------------------------------------------------------
// Kilogram receipt (NOTA DE ENTRADA) processing
// -----------------------------------------------------------------------------

async function extractKilogramWithVision(imageBuffer) {
    if (!OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured");
    }

    const base64 = imageBuffer.toString("base64");

    const response = await openai.chat.completions.create({
        model: OPENAI_MODEL,
        temperature: 0,
        max_tokens: 800,
        response_format: { type: "json_object" },
        messages: [
            {
                role: "system",
                content:
                    "You extract data from Recicladora TAGA NOTA DE ENTRADA receipts. " +
                    "Return JSON only with this exact shape:\n" +
                    "{\n" +
                    '  "serie": string|null,\n' +
                    '  "folio": string|null,\n' +
                    '  "ticket": string|null,\n' +
                    '  "sitio": string|null,\n' +
                    '  "fecha": string|null,\n' +
                    '  "empresa": string|null,\n' +
                    '  "matricula": string|null,\n' +
                    '  "chofer": string|null,\n' +
                    '  "entrada": string|null,\n' +
                    '  "salida": string|null,\n' +
                    '  "movimiento": string|null,\n' +
                    '  "producto": string|null,\n' +
                    '  "bruto": number|null,\n' +
                    '  "tara": number|null,\n' +
                    '  "neto": number|null,\n' +
                    '  "costo": number|null,\n' +
                    '  "total": number|null,\n' +
                    '  "pesador": string|null,\n' +
                    '  "recibido": boolean\n' +
                    "}\n" +
                    "Rules:\n" +
                    "- sitio is the location line near the top (e.g. SOLUCIONES PATIO VIAS).\n" +
                    "- serie is the letter after SERIE (e.g. A). folio/ticket are the receipt number (e.g. 15736).\n" +
                    "- fecha, entrada, salida: keep date and time, prefer YYYY-MM-DD HH:mm:ss.\n" +
                    "- bruto, tara, neto, costo, total: plain numbers (20060.00 -> 20060 or 20060.00).\n" +
                    "- recibido: true if a RECIBIDO stamp/signature is present, else false.\n" +
                    "- If a value is missing or unreadable, return null (or false for recibido).\n" +
                    "- Do not invent values. Never apologize."
            },
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Extract all fields from this NOTA DE ENTRADA image."
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${base64}`,
                            detail: "high"
                        }
                    }
                ]
            }
        ]
    });

    const text = response.choices?.[0]?.message?.content || "{}";

    try {
        return JSON.parse(text);
    } catch {
        throw new Error(
            "OpenAI returned invalid JSON for kilogram extraction"
        );
    }
}

function buildKilogramResponse(raw) {
    const folio =
        cleanFolio(raw?.folio) ||
        cleanFolio(raw?.ticket) ||
        null;

    return {
        serie: cleanOcrText(raw?.serie) || null,
        folio,
        ticket: cleanFolio(raw?.ticket) || folio,
        sitio: cleanOcrText(raw?.sitio) || null,
        fecha: parseDateTime(raw?.fecha) || null,
        empresa: cleanOcrText(raw?.empresa) || null,
        matricula: cleanOcrText(raw?.matricula) || null,
        chofer: cleanOcrText(raw?.chofer) || null,
        entrada: parseDateTime(raw?.entrada) || null,
        salida: parseDateTime(raw?.salida) || null,
        movimiento: cleanOcrText(raw?.movimiento) || null,
        producto: cleanOcrText(raw?.producto) || null,
        bruto: parseWeight(raw?.bruto),
        tara: parseWeight(raw?.tara),
        neto: parseWeight(raw?.neto),
        costo: parseWeight(raw?.costo),
        total: parseWeight(raw?.total),
        pesador: cleanOcrText(raw?.pesador) || null,
        recibido: Boolean(raw?.recibido)
    };
}

async function processKilogramDocument(inputBuffer) {
    if (!cv) {
        return processKilogramVisionOnly(inputBuffer);
    }

    if (!fs.existsSync(KILOGRAM_TEMPLATE_PATH)) {
        throw new Error(
            `Template file not found: ${KILOGRAM_TEMPLATE_PATH}`
        );
    }

    let imageForOcr;

    try {
        const aligned = alignDocument(inputBuffer, {
            templatePath: KILOGRAM_TEMPLATE_PATH,
            width: KILOGRAM_W,
            height: KILOGRAM_H,
            orientation: "portrait"
        });
        imageForOcr = aligned.color;
        console.log("Kilogram receipt aligned to template");
    } catch (error) {
        console.warn(
            "Kilogram alignment failed, using orientation-normalized original:",
            error.message
        );

        let { color } = loadImage(inputBuffer);
        color = rotateToPortrait(color);
        imageForOcr = color;
    }

    const jpegBuffer = matToJpeg(imageForOcr);
    const extracted = await extractKilogramWithVision(jpegBuffer);

    return buildKilogramResponse(extracted);
}

function normalizeProcessFlag(flag) {
    const value = String(flag ?? "sheet")
        .trim()
        .toLowerCase();

    if (
        value === "kilogram" ||
        value === "kilograms" ||
        value === "kg" ||
        value === "peso"
    ) {
        return "kilogram";
    }

    if (
        !value ||
        value === "sheet" ||
        value === "orden" ||
        value === "salida"
    ) {
        return "sheet";
    }

    return null;
}

// -----------------------------------------------------------------------------
// API
// -----------------------------------------------------------------------------

app.post("/process", async (req, res) => {
    try {
        const { image, flag } = req.body;

        if (!image) {
            return res.status(400).json({
                error: "Missing 'image' field"
            });
        }

        const processFlag = normalizeProcessFlag(flag ?? "sheet");

        if (!processFlag) {
            return res.status(400).json({
                error: "Invalid 'flag'. Use 'sheet' or 'kilogram'."
            });
        }

        let base64 = image;

        if (base64.includes(",")) {
            base64 = base64.split(",")[1];
        }

        const inputBuffer = Buffer.from(base64, "base64");

        if (!inputBuffer.length) {
            return res.status(400).json({
                error: "Invalid base64 image"
            });
        }

        const result =
            processFlag === "kilogram"
                ? await processKilogramDocument(inputBuffer)
                : await processDocument(inputBuffer);

        return res.json(result);
    } catch (error) {
        console.error(error);

        return res.status(500).json({
            error: error.message
        });
    }
});

async function start() {
    if (!OPENAI_API_KEY) {
        throw new Error(
            "OPENAI_API_KEY environment variable is required"
        );
    }

    console.log(`Using OpenAI Vision model: ${OPENAI_MODEL}`);

    app.listen(PORT, () => {
        console.log(`OCR server listening on port ${PORT}`);
    });
}

async function shutdown() {
    console.log("Shutting down...");
    process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

start().catch((error) => {
    console.error("Failed to start server:", error);
    process.exit(1);
});
