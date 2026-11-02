import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, stat, rename } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root=fileURLToPath(new URL('../',import.meta.url));
const lock=JSON.parse(await readFile(path.join(root,'assets.lock.json'),'utf8'));
async function hash(file){const h=createHash('sha256');for await(const chunk of createReadStream(file))h.update(chunk);return h.digest('hex');}
async function matches(file,bytes,sha){try{return (await stat(file)).size===bytes&&await hash(file)===sha;}catch{return false;}}
async function check(){let bad=0;for(const f of lock.files){if(!f.path.startsWith('public/assets/')||f.path.includes('..'))throw Error('Invalid manifest path');if(!await matches(path.join(root,f.path),f.bytes,f.sha256)){console.log('Missing or changed: '+f.path);bad++;}}return bad===0;}
if(await check()){console.log('All terrain assets match the release manifest.');process.exit(0);}
if(process.argv.includes('--check'))process.exit(1);
const cache=path.join(root,'.cache');await mkdir(cache,{recursive:true});
const archive=path.join(cache,`terrain-assets-${lock.version}.tar.gz`);
if(!await matches(archive,lock.bytes,lock.sha256)){
 console.log(`Downloading ${(lock.bytes/1048576).toFixed(0)} MiB from ${lock.url}`);
 const response=await fetch(lock.url);if(!response.ok)throw Error(`Download failed: HTTP ${response.status}`);
 await pipeline(Readable.fromWeb(response.body),createWriteStream(archive+'.part'));
 if(!await matches(archive+'.part',lock.bytes,lock.sha256))throw Error('Asset archive checksum mismatch');
 await rename(archive+'.part',archive);
}
const extracted=spawnSync('tar',['-xzf',archive,'-C',root],{stdio:'inherit'});
if(extracted.error||extracted.status!==0)throw Error('Extraction failed. Install the system tar command.');
if(!await check())throw Error('Extracted assets failed verification');
console.log('Terrain assets installed and verified. Run npm run dev.');
