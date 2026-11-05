// [Kehai audit 2026-11-05] LCS floor: how much do genuinely fresh held-out generator docs overlap the training suite?
// Usage: node tests/audit-v04-lcs.js 16 10   (same seeds and sizes as metrics.js)
const fs=require('fs'),path=require('path'),vm=require('vm');const G=require('./generators.js');
const html=fs.readFileSync(path.join(__dirname,'..','index.html'),'utf8');const s='// === FABULA_ENGINE_START ===',e='// === FABULA_ENGINE_END ===';
globalThis.window=globalThis;globalThis.self=globalThis;vm.runInThisContext(html.slice(html.indexOf(s)+s.length,html.indexOf(e)));const E=globalThis.FabulaEngine;
const NT=+process.argv[2],NE=+process.argv[3];
const tr=G.buildSeededSuite(4242,NT), ho=G.buildSeededSuite(97531,NE);
const r1=G.mulberry32(4242),r2=G.mulberry32(97531);tr.bmp=[];ho.bmp=[];
for(let i=0;i<NT;i++)tr.bmp.push(G.generateBMPSample(r1,{size:32,nShapes:1+(i%3)}));
for(let i=0;i<NE;i++)ho.bmp.push(G.generateBMPSample(r2,{size:32,nShapes:1+(i%3)}));
const lcs=E.longestCommonSubstringRatio||globalThis.longestCommonSubstringRatio;
for(const k of ['html','json','svg','js','c','bmp']){const corp=tr[k].map(d=>d.bytes);const v=ho[k].map(d=>lcs(d.bytes,corp));
console.log(k.padEnd(5),'held-out real doc vs train: mean LCS',(v.reduce((a,b)=>a+b,0)/v.length).toFixed(3),'max',Math.max(...v).toFixed(3));}
const crypto=require('crypto');const h=b=>crypto.createHash('sha1').update(Buffer.from(b)).digest('hex');
for(const k of ['html','json','svg','js','c','bmp']){const S=new Set(tr[k].map(d=>h(d.bytes)));const dup=ho[k].filter(d=>S.has(h(d.bytes))).length;
const uniqTr=S.size;console.log(k.padEnd(5),'held-out docs identical to a training doc:',dup+'/'+NE,' | distinct training docs:',uniqTr+'/'+NT);}
