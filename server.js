import express from 'express';
import sharp from 'sharp';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const CACHE_DIR = '.cache';
if (!existsSync(CACHE_DIR)) await mkdir(CACHE_DIR);

const app = express();
app.use(express.json({ limit: '2mb' }));
app.use(express.static('public'));

function normalizeRotation(raw) {
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n)) return 0;
  return ((n % 360) + 360) % 360;
}

async function fetchAndRotate(imageUrl, rotation) {
  const res = await fetch(imageUrl);
  if (!res.ok) throw new Error(`image fetch failed (${res.status})`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (!rotation) return buf;
  return await sharp(buf).rotate(rotation).jpeg({ quality: 90 }).toBuffer();
}

app.get('/image', async (req, res) => {
  const url = req.query.url;
  const rotation = normalizeRotation(req.query.rotation);
  if (!url) return res.status(400).send('url required');
  try {
    const buf = await fetchAndRotate(url, rotation);
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'public, max-age=300');
    res.send(buf);
  } catch (e) {
    res.status(502).send(e.message);
  }
});

app.post('/ocr', async (req, res) => {
  try {
    const { imageUrl, languageHints, apiKey } = req.body;
    const rotation = normalizeRotation(req.body.rotation);
    if (!apiKey) return res.status(400).json({ error: 'apiKey required' });
    if (!imageUrl) return res.status(400).json({ error: 'imageUrl required' });

    const cacheKey = createHash('sha256').update(`${imageUrl}|${rotation}`).digest('hex');
    const cachePath = path.join(CACHE_DIR, `${cacheKey}.json`);

    if (existsSync(cachePath)) {
      const cached = JSON.parse(await readFile(cachePath, 'utf8'));
      return res.json({ ...cached, cached: true });
    }

    let imageField;
    if (rotation === 0) {
      // Let Vision fetch the original directly — avoids sending a 7MB base64 body.
      imageField = { source: { imageUri: imageUrl } };
    } else {
      // Rotation required: fetch, rotate with sharp, send as base64 content.
      let imageBytes;
      try {
        imageBytes = await fetchAndRotate(imageUrl, rotation);
      } catch (e) {
        return res.status(502).json({ error: e.message });
      }
      imageField = { content: imageBytes.toString('base64') };
    }

    const visionRes = await fetch(
      `https://vision.googleapis.com/v1/images:annotate?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requests: [{
            image: imageField,
            features: [{ type: 'DOCUMENT_TEXT_DETECTION' }],
            imageContext: { languageHints: languageHints || ['uk', 'en'] }
          }]
        })
      }
    );

    let visionData;
    try {
      visionData = await visionRes.json();
    } catch (e) {
      return res.status(visionRes.status).json({
        error: `Vision returned non-JSON (status ${visionRes.status})`
      });
    }

    if (visionData.error) {
      return res.status(visionRes.status || 500).json({ error: visionData.error });
    }
    const annotation = visionData.responses?.[0];
    if (annotation?.error) {
      return res.status(500).json({ error: annotation.error });
    }

    const words = extractWords(annotation);
    const result = { words };
    await writeFile(cachePath, JSON.stringify(result));
    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

function extractWords(annotation) {
  const words = [];
  const pages = annotation?.fullTextAnnotation?.pages || [];
  for (const page of pages) {
    for (const block of page.blocks || []) {
      for (const para of block.paragraphs || []) {
        for (const word of para.words || []) {
          const symbols = word.symbols || [];
          const text = symbols.map(s => s.text).join('');
          const verts = word.boundingBox?.vertices || [];
          if (verts.length < 4 || !text) continue;
          const xs = verts.map(v => v.x || 0);
          const ys = verts.map(v => v.y || 0);
          const x = Math.min(...xs);
          const y = Math.min(...ys);
          const w = Math.max(...xs) - x;
          const h = Math.max(...ys) - y;
          if ((word.confidence ?? 1) < 0.5) continue;
          const breakType = symbols[symbols.length - 1]?.property?.detectedBreak?.type;
          const trail = breakType === 'LINE_BREAK' || breakType === 'EOL_SURE_SPACE'
            ? '\n'
            : breakType ? ' ' : '';
          words.push({ text: text + trail, x, y, w, h });
        }
      }
    }
  }
  return words;
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`POC running at http://localhost:${PORT}`));
