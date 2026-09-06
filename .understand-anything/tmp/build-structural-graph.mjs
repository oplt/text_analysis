#!/usr/bin/env node
/**
 * Deterministic structural graph builder for /understand.
 * Runs extract-structure.mjs per batch and converts results to batch-*.json graph format.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = process.argv[2] || process.cwd();
const SKILL_DIR = process.argv[3] || join(dirname(fileURLToPath(import.meta.url)), '../../.agents/skills/understand');
const INTER = join(PROJECT_ROOT, '.understand-anything/intermediate');
const TMP = join(PROJECT_ROOT, '.understand-anything/tmp');

const batches = JSON.parse(readFileSync(join(INTER, 'batches.json'), 'utf8'));
const scan = JSON.parse(readFileSync(join(INTER, 'scan-result.json'), 'utf8'));

function prefixForCategory(cat) {
  const map = {
    config: 'config',
    docs: 'document',
    document: 'document',
    infra: 'service',
    data: 'schema',
    markup: 'document',
  };
  return map[cat] || 'file';
}

function summarize(path, cat, metrics = {}) {
  const parts = path.split('/');
  const name = parts.at(-1) || path;
  if (path.includes('backend/modules/')) {
    const mod = parts[2] ?? 'module';
    return `${mod} module file: ${name}`;
  }
  if (path.startsWith('frontend/src/')) {
    return `Frontend ${cat}: ${name}`;
  }
  if (path.startsWith('backend/api/')) return `API layer: ${name}`;
  if (path.startsWith('backend/core/')) return `Core shared: ${name}`;
  if (path.startsWith('backend/lib/')) return `Shared lib: ${name}`;
  if (path.startsWith('docs/')) return `Documentation: ${name}`;
  if (path.startsWith('infra/') || path.startsWith('observability/')) return `Infrastructure: ${name}`;
  const fn = metrics.functionCount ?? 0;
  const cls = metrics.classCount ?? 0;
  if (fn || cls) return `${name} (${fn} functions, ${cls} classes)`;
  return `${cat} file: ${name}`;
}

function tagsFor(path, cat, language) {
  const tags = [cat, language].filter(Boolean);
  if (path.includes('/tests/') || path.includes('.test.') || path.includes('test_')) tags.push('test');
  if (path.includes('backend/modules/')) tags.push('backend-module');
  if (path.startsWith('frontend/')) tags.push('frontend');
  if (path.includes('/router.') || path.includes('/routes.')) tags.push('api-route');
  if (path.includes('/repository')) tags.push('repository');
  if (path.includes('/service')) tags.push('service');
  return [...new Set(tags)];
}

function layerHint(path) {
  if (path.startsWith('frontend/')) return 'frontend';
  if (path.startsWith('backend/api/') || path.includes('/router.') || path.includes('/routes.')) return 'api';
  if (path.startsWith('backend/modules/') && (path.includes('/domain/') || path.includes('/models'))) return 'domain';
  if (path.startsWith('backend/modules/') && path.includes('/infrastructure/')) return 'infrastructure';
  if (path.startsWith('backend/modules/') && (path.includes('/application/') || path.includes('/service'))) return 'application';
  if (path.startsWith('backend/core/') || path.startsWith('backend/lib/')) return 'core';
  if (path.startsWith('backend/workers/')) return 'workers';
  if (path.startsWith('docs/')) return 'docs';
  if (path.startsWith('infra/') || path.startsWith('observability/')) return 'infra';
  if (path.startsWith('backend/alembic/')) return 'database';
  return 'other';
}

function convertBatch(batchIndex, batch) {
  const inputPath = join(TMP, `ua-file-analyzer-input-${batchIndex}.json`);
  const extractPath = join(TMP, `ua-file-extract-results-${batchIndex}.json`);
  const input = {
    projectRoot: PROJECT_ROOT,
    batchFiles: batch.files,
    batchImportData: batch.batchImportData || {},
  };
  writeFileSync(inputPath, JSON.stringify(input));

  const result = spawnSync(
    'node',
    [join(SKILL_DIR, 'extract-structure.mjs'), inputPath, extractPath],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) {
    console.error(`Batch ${batchIndex} extract failed:`, result.stderr);
    return { nodes: [], edges: [] };
  }
  if (!existsSync(extractPath)) {
    console.error(`Batch ${batchIndex}: missing extract output`);
    return { nodes: [], edges: [] };
  }

  const extracted = JSON.parse(readFileSync(extractPath, 'utf8'));
  const nodes = [];
  const edges = [];
  const nodeIds = new Set();

  for (const file of extracted.results || []) {
    const cat = file.fileCategory || 'code';
    const prefix = prefixForCategory(cat);
    const fileId = `${prefix}:${file.path}`;
    if (!nodeIds.has(fileId)) {
      nodeIds.add(fileId);
      nodes.push({
        id: fileId,
        type: prefix === 'file' ? 'file' : prefix,
        name: file.path.split('/').pop(),
        filePath: file.path,
        summary: summarize(file.path, cat, file.metrics),
        tags: tagsFor(file.path, cat, file.language),
        complexity: (file.totalLines || 0) > 400 ? 'complex' : (file.totalLines || 0) > 150 ? 'moderate' : 'simple',
        language: file.language,
        layerHint: layerHint(file.path),
      });
    }

    const imports = batch.batchImportData?.[file.path] || scan.importMap?.[file.path] || [];
    for (const target of imports) {
      const targetCat = scan.files.find((f) => f.path === target)?.fileCategory || 'code';
      const targetPrefix = prefixForCategory(targetCat);
      const targetId = `${targetPrefix}:${target}`;
      edges.push({ source: fileId, target: targetId, type: 'imports', weight: 0.7 });
    }

    for (const fn of file.functions || []) {
      const fnId = `function:${file.path}:${fn.name}`;
      if (!nodeIds.has(fnId)) {
        nodeIds.add(fnId);
        nodes.push({
          id: fnId,
          type: 'function',
          name: fn.name,
          filePath: file.path,
          summary: `Function ${fn.name} in ${file.path}`,
          tags: ['function', file.language].filter(Boolean),
          complexity: (fn.endLine - fn.startLine) > 80 ? 'complex' : 'moderate',
        });
        edges.push({ source: fileId, target: fnId, type: 'contains', weight: 1.0 });
      }
    }

    for (const cls of file.classes || []) {
      const clsId = `class:${file.path}:${cls.name}`;
      if (!nodeIds.has(clsId)) {
        nodeIds.add(clsId);
        nodes.push({
          id: clsId,
          type: 'class',
          name: cls.name,
          filePath: file.path,
          summary: `Class ${cls.name} in ${file.path}`,
          tags: ['class', file.language].filter(Boolean),
          complexity: (cls.endLine - cls.startLine) > 120 ? 'complex' : 'moderate',
        });
        edges.push({ source: fileId, target: clsId, type: 'contains', weight: 1.0 });
      }
      for (const method of cls.methods || []) {
        const mId = `function:${file.path}:${cls.name}.${method}`;
        if (!nodeIds.has(mId)) {
          nodeIds.add(mId);
          nodes.push({
            id: mId,
            type: 'function',
            name: `${cls.name}.${method}`,
            filePath: file.path,
            summary: `Method ${method} on ${cls.name}`,
            tags: ['method', 'class', file.language].filter(Boolean),
            complexity: 'moderate',
          });
          edges.push({ source: clsId, target: mId, type: 'contains', weight: 1.0 });
        }
      }
    }

    for (const ep of file.endpoints || []) {
      const epName = `${ep.method}-${ep.path}`;
      const epId = `endpoint:${file.path}:${epName}`;
      if (!nodeIds.has(epId)) {
        nodeIds.add(epId);
        nodes.push({
          id: epId,
          type: 'endpoint',
          name: epName,
          filePath: file.path,
          summary: `HTTP ${ep.method} ${ep.path}`,
          tags: ['endpoint', 'api'],
          complexity: 'simple',
        });
        edges.push({ source: fileId, target: epId, type: 'contains', weight: 1.0 });
      }
    }
  }

  console.error(`Batch ${batchIndex}/${batches.totalBatches}: ${nodes.length} nodes, ${edges.length} edges`);
  return { nodes, edges };
}

for (let i = 0; i < batches.batches.length; i++) {
  const batchIndex = batches.batches[i].batchIndex ?? i + 1;
  const graph = convertBatch(batchIndex, batches.batches[i]);
  writeFileSync(join(INTER, `batch-${batchIndex}.json`), JSON.stringify(graph, null, 2));
}

console.error('Structural batch graphs written.');
