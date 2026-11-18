import{_ as e,a as t,c as n,d as r,f as i,g as a,h as o,i as s,l as c,m as l,n as u,o as d,p as f,r as p,s as m,t as h,u as g,v as _,x as v,y}from"./py-g2h9ydwe.js";const b=t=>e(t,`012`);var ee=class{table;constructor(e){this.table=e}score(e,t){let n=Object.hasOwn(this.table,e)?this.table[e]:void 0,r=n&&Object.hasOwn(n,t)?n[t]:void 0;return r===void 0?-14-2*Math.abs(t.length-1):r}align(e,t){let n=t.map(b),r=e.length,i=n.length;if(!i||r>4*i)return null;let a=-0xde0b6b3a7640000,o=Array.from({length:i+1},()=>Array(r+1).fill(a)),s=Array.from({length:i+1},()=>Array(r+1).fill(0));o[0][0]=0;for(let t=1;t<=i;t++){let i=n[t-1],c=o[t-1],l=o[t],u=s[t];for(let t=0;t<=r;t++)for(let n=0;n<=Math.min(4,t);n++)if(c[t-n]>a){let r=c[t-n]+this.score(i,e.slice(t-n,t));r>l[t]&&(l[t]=r,u[t]=n)}}if(o[i][r]<=a)return null;let c=[],l=r;for(let t=i;t>0;t--){let n=s[t][l];c.push(e.slice(l-n,l)),l-=n}return c.reverse()}};const te=`<bos>`,ne=`<eos>`,x=`<sp>`,re=`<unk>`,ie=[`<pad>`,te,ne,x,re],ae=[`,`,`.`,`?`,`!`,`;`,`:`,`"`,`'`,`(`,`)`,`—`,`…`],oe=/* @__PURE__ */ `p.b.t.d.k.ɡ.f.v.θ.ð.s.z.ʃ.ʒ.h.m.n.ŋ.l.ɹ.j.w.tʃ.dʒ.ɾ.ʔ`.split(`.`),se=/* @__PURE__ */ `ə.ɐ.ɚ.ᵻ.ɪ.i.iː.ɛ.æ.ʌ.ɑː.ɔ.ɔː.ʊ.uː.ɜː.eɪ.aɪ.ɔɪ.oʊ.aʊ.iə.oː.ɑːɹ.ɔːɹ.oːɹ.ɪɹ.ɛɹ.ʊɹ.aɪɚ.aɪə.əl.n̩`.split(`.`),ce=[...oe,...se],le=[...ie,...ae,...ce],ue=new Map(le.map((e,t)=>[e,t])),S=new Set(ce),C=new Set(se);var de=class extends Error{};function fe(e){let t=[te],n=[0],r=[];for(let i of e)if(i[0])t[t.length-1]!==i[1]&&(t.push(i[1]),n.push(0));else for(let e of i[1]){let i=t[t.length-1];(S.has(i)||i===`<unk>`)&&(t.push(x),n.push(0));for(let[i,a]of e)S.has(i)?(t.push(i),n.push(a)):(r.push(i),t.push(re),n.push(0))}if(t.push(ne),n.push(0),!t.some(e=>S.has(e)||e===`<unk>`))throw new de(`the text has nothing to pronounce`);return{symbols:t,ids:t.map(e=>ue.get(e)),stress:n,unknown:r}}const w=new Set(`AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW`.split(` `)),pe={B:`b`,CH:`tʃ`,D:`d`,DH:`ð`,F:`f`,G:`ɡ`,HH:`h`,JH:`dʒ`,K:`k`,L:`l`,M:`m`,N:`n`,NG:`ŋ`,P:`p`,R:`ɹ`,S:`s`,SH:`ʃ`,T:`t`,TH:`θ`,V:`v`,W:`w`,Y:`j`,Z:`z`,ZH:`ʒ`},me={AA:`ɑː`,AE:`æ`,AH:`ʌ`,AO:`ɔː`,AW:`aʊ`,AY:`aɪ`,EH:`ɛ`,ER:`ɜː`,EY:`eɪ`,IH:`ɪ`,IY:`iː`,OW:`oʊ`,OY:`ɔɪ`,UH:`ʊ`,UW:`uː`},he=/* @__PURE__ */ new Set([...Object.keys(pe),...[...w].flatMap(e=>[`${e}0`,`${e}1`,`${e}2`])]),ge=/* @__PURE__ */ new Set([`S`,`Z`,`SH`,`ZH`,`CH`,`JH`]),T=`aeiouy`,_e=[`forc`,`ford`,`sword`,`worn`,`torn`,`sworn`,`borne`,`shorn`,`pork`,`divorc`],ve=[`dis`,`mis`,`un`,`in`,`re`,`non`,`over`,`pre`],ye=[`be`,`de`,`re`,`pre`],be=[`ing`,`ings`,`ingly`,`able`,`y`,`er`,`ers`,`est`],xe=(e,t)=>e.length===1&&t.includes(e),Se=e=>e.length>0&&e.every(e=>he.has(e)),Ce=e=>[...e].filter(e=>T.includes(e)).join(``);function E(e){let t=[];for(let n of y(e)){let e=n[0]===`ˈ`?1:n[0]===`ˌ`?2:0,r=e?n.slice(1):n;if(!S.has(r))throw Error(`'${r}' is not a phone of the inventory`);if(e&&!C.has(r))throw Error(`stress mark on the consonant '${r}'`);t.push([r,e])}return t}var we=class{bases;stress;aligned;runs;starts=[];count;word;constructor(e,t,n){this.word=e,this.bases=t.map(b),this.stress=t.map(e=>xe(e.slice(-1),`012`)?Number(e.slice(-1)):null),t.forEach((e,t)=>{let n=this.bases[t];if(!(n in pe)&&!w.has(n))throw Error(`'${e}' is not an ARPABET symbol`)}),this.aligned=n!==null,this.runs=n===null?t.map(()=>``):[...n];let r=0;for(let e of this.runs)this.starts.push(r),r+=e.length;this.count=this.bases.length}base(e){return e>=0&&e<this.count?this.bases[e]:null}vowel(e){return e>=0&&e<this.count&&w.has(this.bases[e])}unstressed(e){return this.vowel(e)&&!this.stress[e]}letters(e){return Ce(this.runs[e])}before(e){return this.aligned?this.word.slice(0,this.starts[e]):``}after(e){return this.aligned?this.word.slice(this.starts[e]+this.runs[e].length):``}endsWith(e,...t){let n=this.bases.slice(e+1);return n.length===t.length&&n.every((e,n)=>e===t[n])}firstVowel(e){return!this.bases.slice(0,e).some(e=>w.has(e))}nextVowelStress(e){for(let t=e+1;t<this.count;t++)if(w.has(this.bases[t]))return this.stress[t];return null}weakPrefix(e){return this.aligned&&this.firstVowel(e)&&this.letters(e)===`e`&&ye.some(e=>this.word.startsWith(e))&&[`b`,`d`,`r`,`pr`].includes(this.before(e))}};function Te(e,t,n){if(!e.aligned)return`ɔː`;let r=e.letters(t);if([`oa`,`oo`,`ou`].includes(r))return`oː`;if(r!==`o`)return`ɔː`;let i=e.runs[t],a=(i.slice(i.indexOf(`o`)+1)+e.after(t)).slice(1);return n?a.startsWith(`r`)?`ɔː`:[`y`,`ies`,`ied`,`a`,`as`,`ah`,`er`,`ers`].includes(a)||[`ia`,`io`,`iu`,`ing`,`age`].some(e=>a.startsWith(e))?`oː`:`ɔː`:a.startsWith(`e`)||e.before(t).endsWith(`p`)&&a.startsWith(`t`)||_e.some(t=>e.word.includes(t))?`oː`:`ɔː`}function Ee(e,t){let n=e.base(t+1),r=e.base(t+2);return n===`F`&&r===`AH`&&e.base(t+3)===`L`?`i`:n===`T`&&r===`IY`||n===`B`&&(r===`AH`||r===`L`)||n===`F`&&r===`AY`||n===`T`&&e.endsWith(t,`T`)&&[`R`,`ER`].includes(e.base(t-1)??``)&&e.word.endsWith(`ite`)?`ᵻ`:`ɪ`}function De(e,t){let n=e.base(t+1),r=e.base(t-1);return n===`T`&&(e.endsWith(t,`T`)||e.endsWith(t,`T`,`S`))?r===`L`?`ə`:r===`R`?`ᵻ`:`ɪ`:n===`S`&&(e.endsWith(t,`S`,`T`)||e.endsWith(t,`S`,`T`,`L`,`IY`))||n===`JH`?`ɪ`:n===`K`&&e.base(t+2)===`T`?`ɛ`:n===`D`&&e.endsWith(t,`D`)?`ɪ`:`ə`}function Oe(e,t){if(t+1>=e.count||e.aligned&&!(e.runs[t]+e.runs[t+1]).includes(`e`))return!1;let n=e.base(t+1),r=e.base(t-1)??``;return n===`D`&&(r===`T`||r===`D`)&&(e.endsWith(t,`D`)||e.endsWith(t,`D`,`L`,`IY`))?!0:n===`Z`&&ge.has(r)&&e.endsWith(t,`Z`)}function ke(e,t){if(!e.aligned||!e.firstVowel(t)||!e.nextVowelStress(t))return null;let n=e.word;if(t===0&&n.startsWith(`e`)){if(n.startsWith(`ex`))return`ɛ`;let e=n.slice(1,2);return[`n`,`m`,`s`].includes(e)&&n.slice(2,3)&&!T.includes(n[2])&&n[2]!==n[1]?`ɛ`:`ɪ`}return e.weakPrefix(t)?n.startsWith(`pre`)||n.startsWith(`be`)&&[`c`,`g`].includes(n.slice(2,3))?`ɪ`:`ᵻ`:e.letters(t)===`i`&&n.startsWith(`di`)&&n.length>3&&!T.includes(n[2])&&n[2]!==`s`&&T.includes(n[3])?`ᵻ`:null}function Ae(e,t){let n=e.bases[t],r=e.letters(t),i=e.base(t+1),a=e.base(t-1);if(n===`IH`&&i===`NG`)return`ɪ`;if(Oe(e,t))return`ᵻ`;if(n===`AH`){if(a===`Y`&&(e.runs[t-1]+e.runs[t]).includes(`u`))return`ʊ`;if(t===0&&e.word.startsWith(`a`))return`ɐ`;if(t===0&&i===`N`&&e.word.startsWith(`un`))return`ʌ`;if(r===`a`&&ve.includes(e.before(t))&&e.runs[t].startsWith(`a`))return`ɐ`}return ke(e,t)||(r===`e`&&(i===`K`||i===`G`)&&e.runs[t+1].startsWith(`x`)&&e.nextVowelStress(t)?`ɛ`:[`i`,`ai`,`ui`,`y`].includes(r)?Ee(e,t):r===`e`?De(e,t):r===`a`?i===`JH`||i===`N`&&e.base(t+2)===`JH`||i===`S`&&e.endsWith(t,`S`)&&e.word.endsWith(`ace`)||n===`IH`&&!(i===`T`&&e.endsWith(t,`T`))?`ɪ`:`ə`:n===`AH`&&i===`M`&&e.endsWith(t,`M`)&&e.word.endsWith(`ome`)?`ʌ`:n===`IH`&&![`o`,`u`,`ou`].includes(r)?`ɪ`:`ə`)}function je(e,t){let n=e.base(t+1);return n===null||e.endsWith(t,`Z`)||e.endsWith(t,`D`)?[`i`,0]:e.vowel(t+1)?[n===`IH`&&e.base(t+2)===`NG`?`i`:`ɪ`,0]:e.weakPrefix(t)?e.nextVowelStress(t)===1?[`ᵻ`,0]:[`iː`,2]:[`ɪ`,0]}function Me(e,t,n=null){let r=new we(e,t,n),i=[],a=(e,t=0)=>{i.push([e,t||0])},o=0;for(;o<r.count;){let e=r.bases[o],t=r.stress[o]||0,n=r.base(o+1);if(!w.has(e)){if(e===`Y`&&n===`AH`&&r.unstressed(o+1)&&[`N`,`L`].includes(r.base(o-1)??``)&&r.base(o+2)!==`L`&&(!r.aligned||(r.runs[o]+r.runs[o+1]).includes(`i`))){a(`iə`),o+=2;continue}a(pe[e]),o+=1;continue}let s=r.vowel(o+2);if(e===`ER`)a(t?`ɜː`:`ɚ`,t),n===`R`&&(o+=1),r.vowel(o+1)&&a(`ɹ`);else if(n===`R`&&e===`AW`)a(`aʊ`,t),a(`ɚ`),s&&a(`ɹ`),o+=1;else if(n===`R`&&[`AA`,`AO`,`IH`,`IY`,`EH`,`UH`,`UW`,`AY`].includes(e)){let n,i;if(e===`AA`)n=`ɑː`,i=`ɑːɹ`;else if(e===`AO`)n=Te(r,o,s),i=n+`ɹ`;else if(e===`IH`||e===`IY`)n=t&&[`e`,`eo`].includes(r.letters(o))?`iə`:`ɪ`,i=`ɪɹ`;else if(e===`EH`){let e=(r.runs[o+1]+r.after(o+1)).startsWith(`rr`)||r.runs[o].endsWith(`r`)&&r.runs[o+1].startsWith(`r`);n=r.letters(o)===`a`&&e?`æ`:`ɛ`,i=`ɛɹ`}else e===`UH`||e===`UW`?n=i=`ʊɹ`:(n=!r.aligned||be.includes(r.after(o+1))?`aɪɚ`:`aɪ`,i=`aɪɚ`);s?(a(n,t),a(`ɹ`)):a(i,t),o+=1}else if(e===`AY`&&n===`ER`&&r.unstressed(o+1)&&r.runs[o+1]===`er`&&r.runs[o]!==`i`)a(`aɪ`,t);else if(e===`AY`&&n===`ER`&&r.unstressed(o+1))a(`aɪɚ`,t),o+=1,r.vowel(o+1)&&a(`ɹ`);else if(e===`AY`&&n===`AH`&&r.unstressed(o+1))a(`aɪə`,t),o+=1;else if(e===`IY`&&n===`AH`&&r.unstressed(o+1)&&!r.weakPrefix(o)){let e=r.base(o+2);t!==1&&([`L`,`D`,`T`].includes(e??``)||e===`N`&&r.base(o+3)===`S`)?a(`ɪ`):(a(`iə`,t),o+=1)}else!t&&e===`AH`&&n===`N`&&r.base(o-1)===`T`&&r.base(o+2)!==`T`&&i.length>=2&&i[i.length-1][0]===`t`&&C.has(i[i.length-2][0])&&i[i.length-2][1]?(i[i.length-1][0]=`ʔ`,a(`n̩`),o+=1):!t&&e===`AH`&&n===`W`&&r.base(o+2)===`AH`&&r.unstressed(o+2)&&(r.runs[o]+r.runs[o+1]).includes(`u`)?(a(`uː`),o+=1):!t&&e===`AH`&&n===`L`&&o>0&&!(r.vowel(o+2)&&r.stress[o+2])?r.letters(o)===`i`&&r.vowel(o+2)?a(`ɪ`):r.base(o-1)===`Y`&&(r.runs[o-1]+r.runs[o]).includes(`u`)?a(`ʊ`):(a(`əl`),o+=1):!t&&(e===`AH`||e===`IH`)?a(Ae(r,o)):e===`IY`&&(!t||t===2&&n===null)?a(...je(r,o)):!t&&e===`AE`&&o===0&&(r.word.startsWith(`ad`)||r.word.startsWith(`acc`))&&(n===`D`||n===`K`)?a(`ɐ`):(e===`AA`||e===`AO`)&&r.letters(o)===`a`&&(r.before(o).endsWith(`w`)||r.before(o).endsWith(`qu`))&&n===`N`||e===`AO`&&r.letters(o)===`o`?a(`ɔ`,t):a(me[e],t);o+=1}Ne(i);for(let e=0;e<i.length-1;e++)i[e][0]===`n`&&(i[e+1][0]===`k`||i[e+1][0]===`ɡ`)&&(i[e][0]=`ŋ`);return Fe(i),i.map(([e,t])=>[e,t])}function Ne(e){for(let t=1;t<e.length-1;t++){if(e[t][0]!==`t`)continue;let n=e[t-1][0],[r,i]=e[t+1];if(!C.has(n)&&n!==`ɹ`||n===`n̩`||n===`əl`||!C.has(r)||r===`n̩`)continue;let a=r===`oʊ`&&(t+2===e.length||t+3===e.length&&e[t+2][0]===`z`);i&&!a||(r===`ə`||r===`ɚ`)&&t+2<e.length&&e[t+2][0]===`n`||(e[t][0]=`ɾ`)}}function Pe(e){let t=e.map(([e,t])=>[e,t]);return Fe(t),t.map(([e,t])=>[e,t])}function Fe(e){let t=e.flatMap(([e],t)=>C.has(e)?[t]:[]),n=t.filter(t=>e[t][1]===1);for(let t of n.slice(0,-1))e[t][1]=2;t.forEach((n,r)=>{e[n][1]===2&&[r-1,r+1].filter(e=>e>=0&&e<t.length).map(e=>t[e]).some(t=>e[t][1]===1)&&(e[n][1]=0)})}const Ie={"abbreviations.txt":`# Abbreviations written with a full stop, and what to say for them.
#
# Columns (two or more spaces): abbreviation without its stop, when it applies,
# words to say. The stop is not a sentence end. "when":
#   always     wherever it is written with the stop
#   small      like "always", but not in capitals (CT. and FL. are states)
#   title      before a capitalised word, with or without the stop (Mr Smith)
#   number     only before a number (No. 5, p. 12)
# Decided in the code: "St." and "Dr." are Saint or Doctor before a name and
# Street or Drive after one; "est." is "established" before a year and
# "estimated" otherwise; "no." is also "number" after "ref.", "serial",
# "account" and the like; "max.", "min." and "est." in small letters are the
# abbreviation only before a word in small letters ("max. load").

mr        title    mister
mrs       title    mrs
ms        title    ms
prof      title    professor
gen       title    general
col       title    colonel
lt        title    lieutenant
capt      title    captain
sgt       title    sergeant
cpl       title    corporal
rev       title    reverend
hon       title    honorable
gov       title    governor
sen       title    senator
rep       title    representative
pres      title    president
mt        title    mount
ft        title    fort
fr        title    father
jr        always   junior
sr        always   senior
vs        always   versus
etc       always   et cetera
inc       always   incorporated
ltd       always   limited
co        always   company
corp      always   corporation
ave       always   avenue
blvd      always   boulevard
rd        always   road
ln        always   lane
hwy       always   highway
apt       always   apartment
bldg      always   building
dept      always   department
govt      always   government
univ      always   university
assn      always   association
bros      always   brothers
approx    always   approximately
est       always   established
misc      always   miscellaneous
tel       always   telephone
max       always   maximum
min       always   minimum
sq        always   square
pkwy      always   parkway
appt      always   appointment
qty       always   quantity
acct      always   account
avg       always   average
incl      always   including
excl      always   excluding
attn      always   attention
fl        small    floor
ct        small    court
no        number   number
nos       number   numbers
p         number   page
pp        number   pages
fig       number   figure
vol       number   volume
ch        number   chapter
sec       number   section
ref       number   reference
ext       number   extension
ste       number   suite
rm        number   room
ca        number   circa
jan       always   January
feb       always   February
mar       always   March
apr       always   April
jun       always   June
jul       always   July
aug       always   August
sep       always   September
sept      always   September
oct       always   October
nov       always   November
dec       always   December
mon       always   Monday
tue       always   Tuesday
tues      always   Tuesday
wed       always   Wednesday
thu       always   Thursday
thur      always   Thursday
thurs     always   Thursday
fri       always   Friday
`,"chat.txt":`# Chat spellings and what to say for them. Edit freely.
#
# Columns (two or more spaces): the token as typed (any letter case), then
#   words to say            btw -> by the way
#   *LETTERS                read letter by letter: lol -> L O L
#   -                       say nothing
# A third column, "alone", keeps a spelling for replies: it is read this way
# only when it is a clause by itself ("k", "k, thanks", "see you, k?"). Next
# to a word or a figure it is what was typed: in "the value of k" and "8 k"
# the letter K.
# Emoticons such as :) and :)) and emoji are removed before this table is used;
# they are never read and never make a pause.

lol       *LOL
lmao      *LMAO
rofl      *ROFL
omg       *OMG
omfg      *OMG
wtf       *WTF
wth       what the heck
idk       I don't know
idc       I don't care
btw       by the way
imo       in my opinion
imho      in my humble opinion
tbh       to be honest
brb       be right back
bbl       be back later
ttyl      talk to you later
gtg       got to go
g2g       got to go
afaik     as far as I know
iirc      if I recall correctly
fyi       *FYI
faq       *FAQ
diy       *DIY
aka       *AKA
eta       *ETA
tba       *TBA
tbd       *TBD
rsvp      *RSVP
asap      *ASAP
smh       *SMH
irl       in real life
dm        *DM
dms       *DM s
thx       thanks
tnx       thanks
ty        thank you
tysm      thank you so much
yw        you're welcome
np        no problem
pls       please
plz       please
ppl       people
msg       message
msgs      messages
pic       pic
pics      pics
u         you
ur        your
r         are
k         okay      alone
kk        okay
ok        okay
nvm       never mind
bc        because
cuz       because
coz       because
tho       though
thru      through
w/        with
w/o       without
b4        before
gr8       great
l8r       later
2day      today
2moro     tomorrow
2nite     tonight
rn        right now
jk        just kidding
ikr       I know right
omw       on my way
hbu       how about you
wyd       what are you doing
fr        for real
ngl       not gonna lie
icl       I can't lie
istg      I swear to god
idek      I don't even know
tbf       to be fair
iykyk     if you know you know
wsg       what's good
gng       gang
pov       *POV
fyp       *FYP
afk       *AFK
npc       *NPC
gg        *GG
haha      ha ha
hahaha    ha ha ha
hahahaha  ha ha ha
hehe      heh heh
xd        -
xoxo      -
n/a       not applicable

# keys of the keyboard
ctrl      control
cmd       command
esc       escape
`,"endings.txt":`# Endings of web addresses and file names, and how to say them. Edit freely.
#
# Columns (two or more spaces): the ending without its dot, then
#   word      said as a word:        example.com -> "example dot com"
#   letters   read letter by letter: notes.txt   -> "notes dot T X T"
#
# A name with a listed ending is always read as a file or an address. An
# ending that is not listed is read letter by letter as well, but only when
# the name shows in another way that it is a file (an underscore or a slash),
# so that a missing space, "the end.Then", is never taken for a file name.
# Short common words ("it", "in", "me", "us") are left out on purpose.

# web addresses
com       word
org       word
net       word
gov       word
info      word
biz       word
app       word
dev       word
co        word
edu       letters
io        letters
ai        letters
sh        letters
ly        letters
tv        letters
fm        letters
gg        letters
xyz       letters
uk        letters
eu        letters
de        letters
fr        letters
es        letters
nl        letters
se        letters
ch        letters
ca        letters
au        letters
jp        letters
cn        letters
ru        letters
br        letters

# documents and data
pdf       letters
txt       letters
csv       letters
tsv       letters
doc       word
docx      letters
xls       letters
xlsx      letters
ppt       letters
pptx      letters
odt       letters
rtf       letters
tex       letters
md        letters
epub      word
json      word
yaml      word
yml       letters
toml      word
xml       letters
html      letters
htm       letters
sql       letters
log       word
conf      word
cfg       letters
ini       letters
env       letters
lock      word

# pictures, sound, video
png       letters
jpg       letters
jpeg      word
gif       word
svg       letters
bmp       letters
tiff      word
psd       letters
webp      letters
heic      letters
mp3       letters
mp4       letters
m4a       letters
wav       word
flac      word
midi      word
mpeg      word
mov       letters
avi       letters
mkv       letters
webm      letters

# programs and archives
exe       letters
dll       letters
apk       letters
dmg       letters
iso       letters
zip       word
tar       word
gz        letters
tgz       letters
rar       letters
jar       letters
bin       word
bat       word
bak       letters
tmp       letters
py        letters
js        letters
ts        letters
jsx       letters
tsx       letters
css       letters
rb        letters
rs        letters
cpp       letters
php       letters
java      word
swift     word
kt        letters
ipynb     letters
`,"exceptions.txt":`# Pronunciations that win over the dictionary. Written by hand.
#
# Two kinds of entry: words CMUdict lacks (names, recent words) and words where
# its first pronunciation is not the usual one in running speech. Either
# CMUdict notation (the conversion rules then apply) or phones with stress marks.
# Columns: word, pronunciation (two or more spaces between them).
#
# This is the place to teach a new word: add a line. A name of several words
# whose rhythm matters takes one pronunciation per word, separated by "|"
# (see "tung tung tung sahur" below); the stress written there is kept.
# CMUdict notation in short: AA father, AE cat, AH cut (AH0 the "a" of sofa),
# AO law, AW cow, AY my, EH bed, ER her, EY day, IH sit, IY see, OW go, OY boy,
# UH book, UW boot; after a vowel 1 = main stress, 2 = lesser stress, 0 = none;
# consonants B CH D DH (this) F G HH JH (joy) K L M N NG P R S SH T TH (thin)
# V W Y Z ZH (vision).

# usual pronunciation differs from CMUdict's first entry
every         EH1 V R IY0
because       B IH0 K AH1 Z
really        ɹ ˈiə l i
real          ɹ ˈiː ə l
thirty        TH ER1 T IY0
houses        HH AW1 Z AH0 Z
enough        ɪ n ˈʌ f
minutes       m ˈɪ n ɪ t s
email         IY1 M EY2 L
emails        IY1 M EY2 L Z
monday        M AH1 N D EY2
tuesday       T UW1 Z D EY2
wednesday     W EH1 N Z D EY2
thursday      TH ER1 Z D EY2
friday        F R AY1 D EY2
saturday      S AE1 T ER0 D EY2
sunday        S AH1 N D EY2
yesterday     Y EH1 S T ER0 D EY2
didn't        d ˈɪ d n t
wasn't        w ˈʌ z n̩ t
cannot        K AE0 N AA1 T
different     D IH1 F R AH0 N T
difference    D IH1 F R AH0 N S
differences   D IH1 F R AH0 N S AH0 Z
sometimes     S AH1 M T AY2 M Z
downtown      D AW1 N T AW2 N
putting       P UH1 T IH0 NG
women         w ˈɪ m ɪ n
maybe         m ˈeɪ b iː
herself       h ɜː s ˈɛ l f
yourself      j oːɹ s ˈɛ l f
yours         j ˈoːɹ z
direct        d ᵻ ɹ ˈɛ k t
directly      d ᵻ ɹ ˈɛ k t l i
direction     d ᵻ ɹ ˈɛ k ʃ ə n
directions    d ᵻ ɹ ˈɛ k ʃ ə n z
director      d ᵻ ɹ ˈɛ k t ɚ
schedule      S K EH1 JH UW0 L
schedules     S K EH1 JH UW0 L Z
scheduled     S K EH1 JH UW0 L D
conference    K AA1 N F R AH0 N S
conferences   K AA1 N F R AH0 N S AH0 Z
polished      P AA1 L IH0 SH T
polishing     P AA1 L IH0 SH IH0 NG
polishes      P AA1 L IH0 SH IH0 Z
address       AH0 D R EH1 S
addresses     AH0 D R EH1 S IH0 Z
government    G AH1 V ER0 N M AH0 N T
governments   G AH1 V ER0 N M AH0 N T S

# recent words CMUdict lacks
wifi          W AY1 F AY2
wi            W AY1
covid         K OW1 V IH0 D
emoji         IH0 M OW1 JH IY0
emojis        IH0 M OW1 JH IY0 Z
selfie        S EH1 L F IY0
selfies       S EH1 L F IY0 Z
blog          B L AO1 G
blogs         B L AO1 G Z
podcast       P AA1 D K AE2 S T
podcasts      P AA1 D K AE2 S T S
smartphone    S M AA1 R T F OW2 N
smartphones   S M AA1 R T F OW2 N Z
bluetooth     B L UW1 T UW2 TH
hashtag       HH AE1 SH T AE2 G
hashtags      HH AE1 SH T AE2 G Z
login         L AO1 G IH2 N
username      Y UW1 Z ER0 N EY2 M
meme          M IY1 M
memes         M IY1 M Z
gif           G IH1 F
jpeg          JH EY1 P EH2 G
json          JH EY1 S AH0 N
linux         L IH1 N AH0 K S
github        G IH1 T HH AH2 B
chatbot       CH AE1 T B AA2 T
chatbots      CH AE1 T B AA2 T S
webcam        W EH1 B K AE2 M
touchscreen   T AH1 CH S K R IY2 N
netflix       N EH1 T F L IH0 K S
spotify       S P AA1 T IH0 F AY2
instagram     IH1 N S T AH0 G R AE2 M
tiktok        T IH1 K T AA2 K
whatsapp      W AH1 T S AE2 P
youtube       Y UW1 T UW2 B
iphone        AY1 F OW2 N
ipad          AY1 P AE2 D
android       AE1 N D R OY2 D
google        G UW1 G AH0 L
uber          UW1 B ER0
tesla         T EH1 S L AH0
okay          OW2 K EY1

# names and rare words the dictionary lacks or has wrong
emmaleigh     EH1 M AH0 L IY0
hermione      HH ER0 M AY1 AH0 N IY0
persephone    P ER0 S EH1 F AH0 N IY0
coeliac       S IY1 L IY0 AE2 K
celiac        S IY1 L IY0 AE2 K
scabies       S K EY1 B IY0 Z
lyssavirus    L IH1 S AH0 V AY2 R AH0 S
provirus      P R OW0 V AY1 R AH0 S
bobsleigh     B AA1 B S L EY2

# sounds people write: said as the sound, not letter by letter
# ("sh", "shh", "hm", "hmm" and "mm" are in the dictionary; the text rules
# read a hiss and a snore, "Ss!" and "Zzz": "ss" is also two letters)
tsk           T IH1 S K
psst          P S T
pfft          P F T
hmph          HH AH1 M F
psh           P SH
pssh          P SH

# words the text rules say: units, file names
milliamp      M IH1 L IY0 AE2 M P
kilocalorie   K IH1 L AH0 K AE2 L ER0 IY0
kilocalories  K IH1 L AH0 K AE2 L ER0 IY0 Z
gigabit       G IH1 G AH0 B IH2 T
gigabits      G IH1 G AH0 B IH2 T S
megabits      M EH1 G AH0 B IH2 T S
kilobits      K IH1 L AH0 B IH2 T S
nginx         EH1 N JH AH0 N EH2 K S
conf          K AA1 N F
yaml          Y AE1 M AH0 L
toml          T AA1 M AH0 L
mpeg          EH1 M P EH2 G
flac          F L AE1 K
epub          IY1 P AH2 B

# internet slang
skibidi       S K IH1 B IH0 D IY0
rizz          R IH1 Z
rizzler       R IH1 Z L ER0
rizzed        R IH1 Z D
gyatt         G Y AE1 T
gyat          G Y AE1 T
fanum         F AE1 N AH0 M
delulu        D IH0 L UW1 L UW0
solulu        S AH0 L UW1 L UW0
bussin        B AH1 S IH0 N
sussy         S AH1 S IY0
yeet          Y IY1 T
simp          S IH1 M P
bruh          B R AH1
brainrot      B R EY1 N R AA2 T
doomscroll    D UW1 M S K R OW2 L
doomscrolling  D UW1 M S K R OW2 L IH0 NG
mewing        M Y UW1 IH0 NG
mog           M AA1 G
mogging       M AA1 G IH0 NG
mogged        M AA1 G D
looksmaxxing  L UH1 K S M AE2 K S IH0 NG
looksmaxing   L UH1 K S M AE2 K S IH0 NG
crashout      K R AE1 SH AW2 T
goated        G OW1 T IH0 D
unc           AH1 NG K
huzz          HH AH1 Z
bruzz         B R AH1 Z
labubu        L AH0 B UW1 B UW0
matcha        M AA1 CH AH0
clanker       K L AE1 NG K ER0
gurt          G ER1 T
pookie        P UH1 K IY0
bestie        B EH1 S T IY0
lowkey        L OW1 K IY2
highkey       HH AY1 K IY2
finna         F IH1 N AH0
boutta        B AW1 T AH0
tryna         T R AY1 N AH0
imma          AY1 M AH0
fomo          F OW1 M OW0
uwu           UW1 W UW0
bae           B EY1
poggers       P AA1 G ER0 Z
kek           K EH1 K
noob          N UW1 B
stonks        S T AA1 NG K S
chungus       CH AH1 NG G AH0 S
amogus        AH0 M OW1 G AH0 S
griddy        G R IH1 D IY0
blud          B L AH1 D
opps          AA1 P S
ragebait      R EY1 JH B EY2 T
parasocial    P EH2 R AH0 S OW1 SH AH0 L
brr           B ER1

# meme names
tung          T UW1 NG
sahur         S AA0 HH UH1 R
tung tung tung sahur  T UW1 NG | T UW2 NG | T UW2 NG | S AA0 HH UH1 R
tralalero     T R AA2 L AH0 L EH1 R OW0
tralala       T R AA2 L AH0 L AA1
tralaleritos  T R AA2 L AH0 L EH0 R IY1 T OW0 S
bombardiro    B AA2 M B AA0 R D IY1 R OW0
crocodilo     K R AA2 K AH0 D IY1 L OW0
cappuccina    K AE2 P AH0 CH IY1 N AH0
ballerino     B AE2 L ER0 IY1 N OW0
patapim       P AA2 T AH0 P IY1 M
lirili        L IH2 R IY0 L IY1
larila        L AA2 R IY0 L AA1
chimpanzini   CH IH2 M P AE0 N Z IY1 N IY0
bananini      B AE2 N AH0 N IY1 N IY0
trippi        T R IH1 P IY0
troppi        T R AA1 P IY0
assassino     AE2 S AH0 S IY1 N OW0
vaca          V AA1 K AH0
saturno       S AH0 T ER1 N OW0
saturnita     S AE2 T ER0 N IY1 T AH0
boneca        B OW0 N EH1 K AH0
ambalabu      AA2 M B AH0 L AH0 B UW1
camelo        K AH0 M EH1 L OW0
bombombini    B AA2 M B AA0 M B IY1 N IY0
gusini        G UW0 Z IY1 N IY0
trulimero     T R UW2 L IH0 M EH1 R OW0
trulicina     T R UW2 L IH0 CH IY1 N AH0
glorbo        G L AO1 R B OW0
fruttodrillo  F R UW2 T OW0 D R IH1 L OW0
burbaloni     B ER2 B AH0 L OW1 N IY0
luliloli      L UW2 L IY0 L OW1 L IY0
baraboom      B AA2 R AH0 B UW1 M
girafa        JH IH0 R AA1 F AH0
celestre      CH EH0 L EH1 S T R EY0
orangutini    AO0 R AE2 NG AH0 T IY1 N IY0
ananasini     AA2 N AH0 N AH0 S IY1 N IY0
udin          UW1 D IH0 N
espressona    EH2 S P R EH0 S OW1 N AH0
cocofanto     K OW2 K OW0 F AA1 N T OW0
elefanto      EH2 L AH0 F AA1 N T OW0
toasterino    T OW2 S T ER0 IY1 N OW0
garama        G AH0 R AA1 M AH0
madundung     M AA0 D UW1 N D UW0 NG
`,"function_words.txt":`# Function words: weak forms and how each word takes sentence stress.
#
# Written by hand. The dictionary gives every word its full, stressed form;
# in running speech these words are unstressed or half stressed, and a few
# have a different vowel. The choices among valid weak forms (for example
# "of" with ʌ, "has" with ɐ) follow the tokens the TTS model was trained on.
#
# Columns (separated by two or more spaces): word, class, form, final form.
#   u    never stressed
#   uf   unstressed; stressed when it is the last word of its clause
#   ur   unstressed; stressed when no stressed word follows in the clause
#   s    secondary stress; primary when no stressed word follows in the clause
#   ss   always secondary stress
#   si   secondary when it opens the clause, otherwise unstressed; primary
#        when no stressed word follows
#   pf   stressed as written; the final form when it closes the clause
# The form is the medial one. The final form is used when the word is
# promoted; without it the marked (or first) vowel takes the primary stress.

# articles, pronouns, determiners
the         u    ð ə
a           u    ɐ
an          u    ɐ n
i           uf   aɪ
you         u    j uː
he          u    h iː
she         u    ʃ iː
it          u    ɪ t
we          u    w iː
they        u    ð eɪ
me          ss   m ˌiː
him         ss   h ˌɪ m
her         u    h ɜː
us          ss   ˌʌ s
them        ss   ð ˌɛ m
my          uf   m aɪ
your        u    j ʊɹ
his         uf   h ɪ z
its         u    ɪ t s
our         ss   ˌaʊ ɚ
their       u    ð ɛɹ
this        uf   ð ɪ s
that        uf   ð æ t
these       uf   ð iː z
those       uf   ð oʊ z
some        ss   s ˌʌ m
any         ss   ˌɛ n i
who         s    h ˌuː
whose       s    h ˌuː z
which       ss   w ˌɪ tʃ
what        si   w ˌʌ t
where       s    w ˌɛɹ
when        si   w ˌɛ n
why         s    w ˌaɪ
how         s    h ˌaʊ
there       uf   ð ɛɹ

# auxiliaries
am          u    æ m
is          u    ɪ z
are         u    ɑːɹ
was         u    w ʌ z
were        u    w ɜː
be          uf   b iː
been        ss   b ˌɪ n
being       s    b ˌiː ɪ ŋ
have        uf   h æ v
has         uf   h ɐ z            h ˈæ z
had         uf   h æ d            h ˌæ d
having      ss   h ˌæ v ɪ ŋ
will        ur   w ɪ l
would       uf   w ʊ d
shall       ss   ʃ ˌæ l
should      s    ʃ ˌʊ d
can         ur   k æ n
could       uf   k ʊ d
might       s    m ˌaɪ t
get         ur   ɡ ɛ t
got         ur   ɡ ɑː t
getting     ss   ɡ ˌɛ ɾ ɪ ŋ
go          s    ɡ ˌoʊ
goes        ur   ɡ oʊ z
going       s    ɡ ˌoʊ ɪ ŋ
gone        ur   ɡ ɔ n
went        uf   w ɛ n t
make        ss   m ˌeɪ k
makes       ss   m ˌeɪ k s
making      ss   m ˌeɪ k ɪ ŋ
made        s    m ˌeɪ d
put         ss   p ˌʊ t
doing       ss   d ˌuː ɪ ŋ
became      ss   b ɪ k ˌeɪ m
become      s    b ɪ k ˌʌ m

# contractions
i'm         u    aɪ m
i'll        u    aɪ l
i've        u    aɪ v
i'd         u    aɪ d
you're      u    j ʊɹ
you'll      u    j uː l
you've      u    j uː v
you'd       u    j uː d
he's        u    h iː z
he'll       u    h iː l
he'd        u    h iː d
she's       u    ʃ iː z
she'll      u    ʃ iː l
she'd       u    ʃ iː d
it's        u    ɪ t s
it'll       ss   ˌɪ ɾ əl
we're       u    w ɪɹ
we'll       u    w iː l
we've       u    w iː v
we'd        u    w iː d
they're     u    ð eɪ ɚ
they'll     u    ð eɪ l
they've     u    ð eɪ v
they'd      u    ð eɪ d
that's      u    ð æ t s
there's     u    ð ɛɹ z
what's      ss   w ˌʌ t s
who's       ss   h ˌuː z
where's     ss   w ˌɛɹ z
how's       ss   h ˌaʊ z
isn't       s    ˌɪ z ə n t
won't       uf   w oʊ n t
couldn't    s    k ˌʊ d ə n t
shouldn't   s    ʃ ˌʊ d ə n t

# prepositions, conjunctions, particles
of          u    ʌ v
to          uf   t ə              t uː
in          uf   ɪ n
on          s    ˌɔ n
at          u    æ t
by          uf   b aɪ
for         u    f ɔːɹ
from        u    f ɹ ʌ m
with        u    w ɪ ð
as          u    æ z
than        u    ð ɐ n
and         uf   æ n d
or          u    ɔːɹ
if          u    ɪ f
but         s    b ˌʌ t
so          s    s ˌoʊ
not         s    n ˌɑː t
up          s    ˌʌ p
down        s    d ˌaʊ n
about       s    ɐ b ˌaʊ t
above       s    ə b ˌʌ v
across      ss   ə k ɹ ˌɑː s
among       ss   ɐ m ˌʌ ŋ
before      s    b ᵻ f ˌoːɹ
behind      s    b ᵻ h ˌaɪ n d
below       s    b ᵻ l ˌoʊ
beneath     ss   b ᵻ n ˌiː θ
beside      s    b ᵻ s ˌaɪ d
between     s    b ᵻ t w ˌiː n
beyond      ss   b ᵻ j ˌɔ n d
into        ss   ˌɪ n t ʊ
onto        ss   ˌɑː n t ʊ
unto        ss   ˌʌ n t ʊ
upon        ss   ə p ˌɑː n
over        s    ˌoʊ v ɚ
under       s    ˌʌ n d ɚ
near        s    n ˌɪɹ
nearly      ss   n ˌɪɹ l i
through     ur   θ ɹ uː
within      s    w ɪ ð ˌɪ n
without     s    w ɪ ð ˌaʊ t
while       s    w ˌaɪ l
though      s    ð ˌoʊ
together    s    t ə ɡ ˌɛ ð ɚ
already     s    ɔː l ɹ ˌɛ d i
whenever    ss   w ɛ n ˌɛ v ɚ
sir         s    s ˌɜː

# numbers: "one" weakens at the end of a clause; the teens are stressed on the
# first syllable before another word and on "teen" at the end
one         pf   w ˈʌ n           w ˌʌ n
thirteen    pf   θ ˈɜː t iː n     θ ɜː t ˈiː n
fourteen    pf   f ˈoːɹ t iː n    f oːɹ t ˈiː n
fifteen     pf   f ˈɪ f t iː n    f ɪ f t ˈiː n
sixteen     pf   s ˈɪ k s t iː n  s ɪ k s t ˈiː n
seventeen   pf   s ˈɛ v ə n t iː n  s ɛ v ə n t ˈiː n
eighteen    pf   ˈeɪ t iː n       eɪ t ˈiː n
nineteen    pf   n ˈaɪ n t iː n   n aɪ n t ˈiː n
`,"homograph_rules.txt":`# Grammar and context rules for a few very common homographs. Written by hand.
#
# They are tried before the trained classifier, in order; the first rule whose
# conditions all hold decides. The training sentences come from Wikipedia,
# where questions, commands and first-person speech are rare, so the cases
# below ("have you read", "will you read", "take a bow") need saying outright.
#
# Columns (two or more spaces): word, sense, conditions separated by ";".
#   aux: ...     the nearest word to the left, skipping adverbs, "not",
#                pronouns and articles, is one of these
#   before: ...  the word just before (adverbs and "not" skipped) is one of these
#   next: ...    the word just after is one of these; </s> = end of sentence
#   near: ...    one of these occurs within six words
#   first        the word opens the sentence
#   joined: ...  a hyphen joins the word to one of these before it ("well-read")

read     past      joined: well widely best better most least little much oft often rarely seldom never half un
read     past      aux: have has had having haven't hasn't hadn't was were been being is are be am got
read     present   aux: to will would can could should shall may might must do does did don't doesn't didn't let let's please cannot can't won't wouldn't couldn't shouldn't
read     present   first

live     vrb       before: i you we they who people to
live     adj       next: music show shows broadcast performance performances concert concerts stream streaming coverage tv television album recording audience action event events demo feed wire ammunition

wound    vrb       aux: has have had having was were been got get tightly
wound    vrb       next: up around round down through back tightly
wound    vrb       before: he she it and then

bow      nou-ship  aux: take took takes taking taken to will would must
bow      nou-knot  near: tie tied tying ties ribbon ribbons hair arrow arrows violin cello fiddle string strings rainbow gift shoelace shoelaces
bow      nou-ship  near: ship ships boat deck stern captain audience curtain applause stage head knee knees waist

bass     corp      near: fish fishing fished caught catch lake river pond sea striped largemouth smallmouth angler trout

lead     nou       near: pipe pipes paint pencil poisoning metal metals bullet bullets zinc copper tin mercury ore

minute   adj       before: is are was were so very extremely too such ; next: </s> , . ! ? and but details detail amount amounts quantities quantity particles differences changes fraction

content  adj-nou-vrb  before: is are was were am be been feel feels felt seem seems seemed quite so very perfectly not ; next: </s> , . ! ? with to in and
`,"letters.txt":`# Names of the letters, for initialisms and single letters.
a    ˈeɪ
b    b ˈiː
c    s ˈiː
d    d ˈiː
e    ˈiː
f    ˈɛ f
g    dʒ ˈiː
h    ˈeɪ tʃ
i    ˈaɪ
j    dʒ ˈeɪ
k    k ˈeɪ
l    ˈɛ l
m    ˈɛ m
n    ˈɛ n
o    ˈoʊ
p    p ˈiː
q    k j ˈuː
r    ˈɑːɹ
s    ˈɛ s
t    t ˈiː
u    j ˈuː
v    v ˈiː
w    d ˈʌ b əl j uː
x    ˈɛ k s
y    w ˈaɪ
z    z ˈiː
`,"phrases.txt":`# Function-word groups spoken as one unit: no word boundary inside, and often
# a weaker form than the words have alone ("for the", "of a", "would have").
#
# Written by hand. Which pairs are joined follows the tokens the TTS model was
# trained on, so that the common cases look the same to it; switch the table
# off (join_phrases=False) to keep a boundary token between all words.
#
# Columns (two or more spaces): words, class, form, final form. Classes as in
# function_words.txt; p = stress exactly as written. A final "the" or "to"
# still takes its form before a vowel (ð ɪ, t ʊ). A form written with "|"
# keeps the word boundaries and only fixes the stress of the pair.

of the          u    ʌ v ð ə
in the          u    ɪ n ð ə
on the          u    ɔ n ð ə
for the         u    f ɚ ð ə
from the        u    f ɹ ʌ m ð ə
with the        u    w ɪ ð ð ə
that the        u    ð æ t ð ə
was the         u    w ʌ z ð ə
of a            u    ə v ə
of an           u    ə v ə n
for a           u    f ɚ ɹ ə
for an          u    f ɚ ɹ ə n
at a            u    æ ɾ ə
was a           u    w ʌ z ɐ
to be           uf   t ə b i              t ə b ˈiː
had to          u    h æ d t ə
i am            u    aɪ ɐ m
i shall         ss   aɪ ʃ ˌɐ l
we shall        ss   w iː ʃ ˌɐ l
had been        u    h ɐ d b ɪ n
have been       u    h ɐ v b ɪ n
has been        p    h ˈæ z b iː n
would have      u    w ʊ d h ɐ v
could have      ss   k ˌʊ d ɐ v
should have     ss   ʃ ˌʊ d ə v
might have      ss   m ˌaɪ t h ɐ v
must have       p    m ˈʌ s t ɐ v
did not         s    d ɪ d n ˌɑː t
do not          s    d uː n ˌɑː t
does not        s    d ʌ z n ˌɑː t
have not        s    h ɐ v n ˌɑː t
there was       s    ð ɛɹ w ˌʌ z
there were      s    ð ɛɹ w ˌɜː
there are       ss   ð ɛ ɹ ˌɑːɹ
out of          ss   ˌaʊ ɾ ə v
at once         p    ɐ t w ˈʌ n s
no one          p    n ˈoʊ w ˈʌ n
that one        p    ð ˈæ t w ˌʌ n
this one        p    ð ˈɪ s w ˌʌ n
that it         ss   ð ˌɐ ɾ ɪ t
that a          ss   ð ˌæ ɾ ə
not a           ss   n ˌɑː ɾ ə
not have        ss   n ˌɑː ɾ ɐ v
of which        p    ʌ v w ˈɪ tʃ
in which        ss   ɪ n w ˌɪ tʃ

# stress of a pair, boundary kept
so much         p    s ˈoʊ | m ˌʌ tʃ
so far          p    s ˈoʊ | f ˌɑːɹ
too much        p    t ˈuː | m ʌ tʃ
no longer       p    n ˌoʊ | l ˈɔ ŋ ɡ ɚ
`,"units.txt":`# Units written after a number: abbreviation, singular, plural.
# Columns separated by two or more spaces. Letter case is ignored, except for
# the units marked "exact" in a fourth column: those are read only in the
# letter case written here ("5 m" is five meters, "5 M" and "5G" are not units).

km      kilometer            kilometers
kg      kilogram             kilograms
cm      centimeter           centimeters
mm      millimeter           millimeters
ml      milliliter           milliliters
mg      milligram            milligrams
kb      kilobyte             kilobytes
mb      megabyte             megabytes
gb      gigabyte             gigabytes
tb      terabyte             terabytes
hz      hertz                hertz
khz     kilohertz            kilohertz
mhz     megahertz            megahertz
ghz     gigahertz            gigahertz
kw      kilowatt             kilowatts
kwh     kilowatt hour        kilowatt hours
lb      pound                pounds
lbs     pound                pounds
oz      ounce                ounces
ft      foot                 feet
mi      mile                 miles
min     minute               minutes
mins    minute               minutes
hr      hour                 hours
hrs     hour                 hours
sec     second               seconds
secs    second               seconds
ms      millisecond          milliseconds
mph     mile per hour        miles per hour
kph     kilometer per hour   kilometers per hour
km/h    kilometer per hour   kilometers per hour
°c      degree Celsius       degrees Celsius
°f      degree Fahrenheit    degrees Fahrenheit
°       degree               degrees
m/s     meter per second     meters per second
yd      yard                 yards
nm      nanometer            nanometers
μm      micrometer           micrometers
μg      microgram            micrograms
mcg     microgram            micrograms
mah     milliamp hour        milliamp hours
hp      horsepower           horsepower
tsp     teaspoon             teaspoons
tbsp    tablespoon           tablespoons
db      decibel              decibels
kcal    kilocalorie          kilocalories
px      pixel                pixels
rpm     revolution per minute    revolutions per minute
bpm     beat per minute      beats per minute
fps     frame per second     frames per second
mpg     mile per gallon      miles per gallon
psi     pound per square inch    pounds per square inch
kbps    kilobit per second   kilobits per second
mbps    megabit per second   megabits per second
gbps    gigabit per second   gigabits per second
yr      year                 years
yrs     year                 years
wk      week                 weeks
wks     week                 weeks

# one letter, or easily taken for something else: only in this letter case
m       meter                meters               exact
g       gram                 grams                exact
l       liter                liters               exact
L       liter                liters               exact
h       hour                 hours                exact
V       volt                 volts                exact
W       watt                 watts                exact
MW      megawatt             megawatts            exact
p       penny                pence                exact
ha      hectare              hectares             exact
`},Le=/\t+| {2,}/,Re=/* @__PURE__ */ new Set([`u`,`uf`,`ur`,`s`,`ss`,`si`,`pf`,`p`]);function D(e){let t=Ie[e];if(t===void 0)throw Error(`no table ${e}`);let n=[];return t.split(/\r\n|\r|\n/).forEach((e,t)=>{let r=e.trim();r&&!r.startsWith(`#`)&&n.push([t+1,r.split(Le).map(e=>e.trim())])}),n}function ze(){let e=/* @__PURE__ */ new Map;for(let[t,n]of D(`function_words.txt`)){if(n.length!==3&&n.length!==4||!Re.has(n[1]))throw Error(`function_words.txt line ${t}: expected word, class, form[, final form]`);e.set(n[0],[n[1],E(n[2]),n.length===4?E(n[3]):null])}return e}function Be(){let e=/* @__PURE__ */ new Map;for(let[t,n]of D(`phrases.txt`)){if(n.length!==3&&n.length!==4||!Re.has(n[1]))throw Error(`phrases.txt line ${t}: expected words, class, form[, final form]`);let r=n[2].split(`|`).map(e=>E(e));if(r.length>1&&(n[1]!==`p`||r.length!==y(n[0]).length))throw Error(`phrases.txt line ${t}: a form with boundaries needs class p and one part per word`);e.set(l(y(n[0])),[n[1],r,n.length===4?E(n[3]):null])}return e}function Ve(e){let t=y(e);return Se(t)?[`arpabet`,t]:[`phones`,E(e)]}function He(e){let t=/* @__PURE__ */ new Map;for(let[n,r]of D(e)){if(r.length!==2)throw Error(`${n}: expected a word and its pronunciation`);y(r[0]).length===1&&t.set(r[0].toLowerCase(),Ve(r[1]))}return t}function Ue(e){let t=/* @__PURE__ */ new Map;for(let[n,r]of D(e)){let e=y(r[0].toLowerCase());if(r.length!==2||e.length<2)continue;let i=r[1].split(`|`).map(e=>e.trim());if(i.length!==e.length)throw Error(`${n}: one pronunciation per word, separated by |`);t.set(l(e),i.map(Ve))}return t}function We(e){return new Map(D(e).map(([,e])=>[e[0],e.length>1?e[1]:``]))}const Ge=new Map([[`TO`,`to`],[`DET`,`the a an this that these those every each no some any another such what which whose`],[`POSS`,`my your his her its our their`],[`SUBJ`,`i you he she it we they who`],[`OBJ`,`me him us them`],[`MODAL`,`will would shall should can could may might must 'll 'd won't can't cannot couldn't shouldn't wouldn't let's please`],[`DO`,`do does did don't doesn't didn't`],[`BE`,`am is are was were be been being isn't aren't wasn't weren't i'm you're we're they're he's she's it's that's there's`],[`HAVE`,`have has had having haven't hasn't hadn't i've you've we've they've`],[`NEG`,`not never`],[`PREP`,`of in on at by for from with into onto over under about after before between through during without within against among around across behind beside near upon toward towards as than like`],[`CONJ`,`and or but nor so yet if because although though while when where unless until since whether`],[`ADV`,`very too quite rather more most less least also just only even still always often sometimes usually again now then here there soon already almost`]].flatMap(([e,t])=>y(t).map(t=>[t,e]))),Ke=/* @__PURE__ */ new Set([`.`,`?`,`!`]);function O(e){if(e===`<s>`||e===`</s>`)return e;let t=[...e][0];if(!n(t))return g(t)?`NUM`:`PUNCT`;let r=e.toLowerCase(),i=Ge.get(r);if(i)return i;for(let[e,t]of[[`'s`,`POS`],[`ly`,`LY`],[`ing`,`ING`],[`ed`,`ED`]])if(r.endsWith(e)&&r.length>e.length+1)return t;return r.endsWith(`s`)&&!r.endsWith(`ss`)&&r.length>3?`S`:f(t)?`CAP`:`W`}function qe(e,t){let r=n=>{let r=t+n;return r>=0&&r<e.length?e[r]:n<0?`<s>`:`</s>`},[i,a,o,s]=[r(-2),r(-1),r(1),r(2)],[c,l,u,d]=[O(i),O(a),O(o),O(s)],p=[`tp1=${l}`,`tp2=${c}`,`tn1=${u}`,`tn2=${d}`,`tp2p1=${c}+${l}`,`tp1n1=${l}+${u}`,`tn1n2=${u}+${d}`];t===0?p.push(`first`):f([...e[t]][0])&&p.push(`cap`),p.push(`p1=${a.toLowerCase()}`,`p2=${i.toLowerCase()}`,`n1=${o.toLowerCase()}`,`n2=${s.toLowerCase()}`);let m=e[t].toLowerCase(),g=new Set(e.slice(Math.max(0,t-6),t+6+1).filter(e=>n([...e][0])).map(e=>e.toLowerCase()));return g.delete(m),p.push(...[...g].sort(h).map(e=>`w=${e}`)),p}const Je=e=>e.filter(e=>e[0]===`t`||e===`first`||e===`cap`);function Ye(e,t){let n=t;for(;n>0&&!Ke.has(e[n-1]);)n--;let r=t;for(;r<e.length&&!Ke.has(e[r]);)r++;return[e.slice(n,r+1),t-n]}const Xe=/* @__PURE__ */ new Set([`ADV`,`NEG`,`LY`]),Ze=/* @__PURE__ */ new Set([...Xe,`SUBJ`,`OBJ`,`DET`,`POSS`]);function Qe(e,t,n){for(let r=t-1;r>Math.max(t-6,-1);r--){let t=e[r];if(n.has(O(t)))continue;let i=t.toLowerCase();return i.endsWith(`'ll`)?`will`:i.endsWith(`'ve`)?`have`:i}return`<s>`}function $e(){let e=/* @__PURE__ */ new Map;for(let[t,n]of D(`homograph_rules.txt`)){if(n.length!==3)throw Error(`homograph_rules.txt line ${t}: expected word, sense, conditions`);let r=[];for(let e of n[2].split(`;`)){let n=e.trim(),i=n.indexOf(`:`),a=i<0?n:n.slice(0,i),o=i<0?``:n.slice(i+1);if(![`aux`,`before`,`next`,`near`,`first`,`joined`].includes(a)||a===`first`==!!o.trim())throw Error(`homograph_rules.txt line ${t}: bad condition '${n}'`);r.push([a,new Set(y(o))])}let i=e.get(n[0])??[];i.push([n[1],r]),e.set(n[0],i)}return e}function et(e,t,n,r,i=!1){return e===`first`?r===0:e===`joined`?i&&r>0&&t.has(n[r-1].toLowerCase()):e===`aux`?t.has(Qe(n,r,Ze)):e===`before`?t.has(Qe(n,r,Xe)):e===`next`?t.has(r+1<n.length?n[r+1].toLowerCase():`</s>`):n.slice(Math.max(0,r-6),r+6+1).some(e=>t.has(e.toLowerCase()))}const tt=(e,t)=>Object.hasOwn(e,t)?e[t]:0;function nt(e,t){let n=0;for(let r of t)n+=tt(e,r);return n}var rt=class{words;pooled;pluralBases;rules;constructor(e){this.words=e.words,this.pooled=e.pooled,this.pluralBases=new Set(e.plural_bases??[]),this.rules=$e();for(let[e,t]of this.rules){let n=new Set(this.has(e)?this.words[e].senses.map(e=>e.id):[]);for(let[r]of t)if(!n.has(r))throw Error(`homograph rule for '${e}' names the unknown sense '${r}'`)}}has(e){return Object.hasOwn(this.words,e)}covers(e){return this.has(e)||e.endsWith(`s`)&&this.pluralBases.has(e.slice(0,-1))}pronounce(e,t,n,r=!1){if(this.has(e))return[this.choose(e,t,n,r).phones,!1];let i=this.words[e.slice(0,-1)],a=i.senses.find(e=>e.role===`n`);return a&&a.id===`adj`?[i.senses.find(e=>e.role===`v`).phones,!0]:[this.choose(e.slice(0,-1),t,n,r).phones,!0]}scores(e,t,n){let r=this.words[e],i=qe(...Ye(t,n)),a=r.model;if(a.kind===`noun-verb`){let e=a,t=this.pooled.bias+nt(this.pooled.weights,Je(i));return t+=e.bias+nt(e.weights,i),r.senses.map(e=>e.role===`v`?t:-t)}let o=a;return o.bias.map((e,t)=>e+nt(o.weights[t],i))}choose(e,t,n,r=!1){let i=this.words[e],[a,o]=Ye(t,n);for(let[t,n]of this.rules.get(e)??[])if(n.every(([e,t])=>et(e,t,a,o,r)))return i.senses.find(e=>e.id===t);let s=this.scores(e,t,n),c=0;return s.forEach((e,t)=>{e>s[c]&&(c=t)}),i.senses[c]}};const it=/(?:^|[^aeiou])[aeiou](?:[bcdfgklmnprstvz]|th)$/,at=/* @__PURE__ */ new Set([`s`,`z`,`ʃ`,`ʒ`,`tʃ`,`dʒ`]),ot=/* @__PURE__ */ new Set([`p`,`t`,`k`,`f`,`θ`,`ʔ`]),st={ɑːɹ:`ɑː`,ɔːɹ:`ɔː`,oːɹ:`oː`,ɪɹ:`ɪ`,ɛɹ:`ɛ`},ct=/* @__PURE__ */ new Set([`ɚ`,`ɜː`,`aɪɚ`,`ʊɹ`]),k=(...e)=>e.map(e=>[e,0]),lt=[[`fully`,k(`f`,`əl`,`i`)],[`lessly`,k(`l`,`ə`,`s`,`l`,`i`)],[`ness`,k(`n`,`ə`,`s`)],[`less`,k(`l`,`ə`,`s`)],[`ments`,k(`m`,`ə`,`n`,`t`,`s`)],[`ment`,k(`m`,`ə`,`n`,`t`)],[`ful`,k(`f`,`əl`)],[`ship`,[[`ʃ`,0],[`ɪ`,2],[`p`,0]]],[`hood`,[[`h`,0],[`ʊ`,2],[`d`,0]]],[`like`,[[`l`,0],[`aɪ`,2],[`k`,0]]],[`dom`,k(`d`,`ə`,`m`)],[`ish`,k(`ɪ`,`ʃ`)],[`able`,k(`ə`,`b`,`əl`)],[`ism`,k(`ɪ`,`z`,`ə`,`m`)],[`ist`,k(`ɪ`,`s`,`t`)],[`wards`,k(`w`,`ɚ`,`d`,`z`)],[`ward`,k(`w`,`ɚ`,`d`)],[`age`,k(`ɪ`,`dʒ`)],[`ers`,k(`ɚ`,`z`)],[`er`,k(`ɚ`)],[`est`,k(`ɪ`,`s`,`t`)],[`y`,k(`i`)]],A=e=>[e,e],ut=[[`under`,A([[`ʌ`,2],[`n`,0],[`d`,0],[`ɚ`,0]])],[`over`,A([[`oʊ`,2],[`v`,0],[`ɚ`,0]])],[`anti`,A([[`æ`,2],[`n`,0],[`t`,0],[`i`,0]])],[`semi`,A([[`s`,0],[`ɛ`,2],[`m`,0],[`i`,0]])],[`non`,A([[`n`,0],[`ɑː`,2],[`n`,0]])],[`out`,A([[`aʊ`,2],[`t`,0]])],[`dis`,[k(`d`,`ɪ`,`s`),[[`d`,0],[`ɪ`,2],[`s`,0]]]],[`mis`,[k(`m`,`ɪ`,`s`),[[`m`,0],[`ɪ`,2],[`s`,0]]]],[`un`,[k(`ʌ`,`n`),[[`ʌ`,2],[`n`,0]]]],[`in`,[k(`ɪ`,`n`),[[`ɪ`,2],[`n`,0]]]],[`re`,[k(`ɹ`,`ᵻ`),[[`ɹ`,0],[`iː`,2]]]],[`pre`,A([[`p`,0],[`ɹ`,0],[`iː`,2]])]],dt=[[`isation`,`ization`],[`ising`,`izing`],[`ised`,`ized`],[`ises`,`izes`],[`ise`,`ize`],[`yse`,`yze`],[`our`,`or`],[`ourable`,`orable`],[`ours`,`ors`],[`oured`,`ored`],[`ouring`,`oring`],[`tre`,`ter`],[`tres`,`ters`],[`bre`,`ber`],[`ence`,`ense`],[`ogue`,`og`],[`mme`,`m`],[`lled`,`led`],[`lling`,`ling`],[`ller`,`ler`],[`oul`,`ol`],[`ae`,`e`],[`enqui`,`inqui`]];function j(e,t){let n=[...e];if(!n.length||!t.length)return[...n,...t];let[r,i]=n[n.length-1],[a,o]=t[0];return C.has(a)?r in st?n.splice(-1,1,[st[r],i],[`ɹ`,0]):ct.has(r)?n.push([`ɹ`,0]):r===`t`&&n.length>1&&(C.has(n[n.length-2][0])||n[n.length-2][0]===`ɹ`)&&!o&&!(a===`ə`&&t.length>1&&t[1][0]===`n`)?n[n.length-1]=[`ɾ`,0]:r===`i`&&(n[n.length-1]=a===`ɪ`?[`i`,i]:[`ɪ`,i]):r===`i`&&[`n`,`l`,`f`,`m`].includes(a)?n[n.length-1]=[`ɪ`,i]:r===`n`&&(a===`k`||a===`ɡ`)&&(n[n.length-1]=[`ŋ`,0]),[...n,...t]}function M(e){let t=e[e.length-1][0];return j(e,at.has(t)?k(`ᵻ`,`z`):ot.has(t)?k(`s`):k(`z`))}function ft(e){let t=e[e.length-1][0];return t===`t`||t===`d`||t===`ɾ`?j(e,k(`ᵻ`,`d`)):j(e,ot.has(t)||[`s`,`ʃ`,`tʃ`].includes(t)?k(`t`):k(`d`))}function pt(e){let t=e[e.length-1][0];return j(e,t===`əl`||t===`l`?k(`i`):k(`l`,`i`))}const mt=e=>e!==void 0&&e.length===1&&`bcdfghjklmnpqrstvwxz`.includes(e);function ht(e){let t=[e],n=e[e.length-1];if(e.length>2&&n===e[e.length-2]&&mt(n)&&t.push(e.slice(0,-1)),e&&(mt(n)||n===`u`)){let n=it.test(e);t.splice(n?0:t.length,0,e+`e`)}return e.endsWith(`i`)&&e.length>2&&t.push(e.slice(0,-1)+`y`),t}const gt=e=>e.map(([e,t])=>[e,t===1?2:t]),_t=(e,t)=>j(e,gt(t));function vt(e,t,n=null,r=0){let i=n??t,a=e=>!!e&&e.length>0,o=e=>{if(e.length<2)return null;let n=t(e);return a(n)?n:r<2?vt(e,t,i,r+1):null},s=(e,t)=>{for(let n of e){let e=o(n);if(a(e))return t(e)}return null};if(e.endsWith(`'s`))return s([e.slice(0,-2)],M);if(e.endsWith(`s'`))return s([e.slice(0,-1)],e=>e);let c;if(e.endsWith(`ies`)&&e.length>4&&(c=s([e.slice(0,-3)+`y`],M),a(c))||e.endsWith(`ves`)&&e.length>4&&(c=s([e.slice(0,-3)+`f`,e.slice(0,-3)+`fe`],e=>e[e.length-1][0]===`f`?j([...e.slice(0,-1),[`v`,0]],k(`z`)):null),a(c))||e.endsWith(`es`)&&e.length>3&&(c=s(`sxzh`.includes(e[e.length-3])?[e.slice(0,-2)]:[],M),a(c))||e.endsWith(`s`)&&![`ss`,`us`,`is`].some(t=>e.endsWith(t))&&e.length>3&&(c=s([e.slice(0,-1)],M),a(c))||e.endsWith(`ied`)&&e.length>4&&(c=s([e.slice(0,-3)+`y`],ft),a(c))||e.endsWith(`ed`)&&e.length>3&&(c=s(ht(e.slice(0,-2)),ft),a(c))||e.endsWith(`ing`)&&e.length>4&&(c=s(ht(e.slice(0,-3)),e=>j(e,k(`ɪ`,`ŋ`))),a(c))||e.endsWith(`ily`)&&e.length>4&&(c=s([e.slice(0,-3)+`y`],pt),a(c))||e.endsWith(`ly`)&&e.length>3&&(c=s(e.endsWith(`bly`)||e.endsWith(`ply`)?[e.slice(0,-2),e.slice(0,-2)+`le`,e.slice(0,-1)+`e`]:[e.slice(0,-2)],pt),a(c)))return c;for(let[t,n]of lt)if(e.endsWith(t)&&e.length-t.length>2){let r=e.slice(0,-t.length);if(t===`y`&&!mt(r[r.length-1]))continue;if(c=s(C.has(n[0][0])?ht(r):r.endsWith(`i`)?[r.slice(0,-1)+`y`,r]:[r],e=>j(e,n)),a(c))return c}for(let[t,[n,r]]of ut)if(e.startsWith(t)&&e.length-t.length>2){let i=o(e.slice(t.length));if(a(i)){let e=i.filter(([e])=>C.has(e)).map(([,e])=>e);return j(e.length&&e[0]===1?n:r,i)}}for(let[t,n]of dt)if(e.includes(t)){let r=o(e.replace(t,n));if(a(r))return r}if(r===0){let t=null;for(let n=4;n<e.length-3;n++){let r=i(e.slice(0,n)),o=i(e.slice(n));if(a(r)&&a(o)){let i=Math.min(n,e.length-n);(t===null||i>t[0])&&(t=[i,r,o])}}if(t)return _t(t[1],t[2])}return null}const N=(e,t={})=>({kind:`word`,text:e,attach:t.attach??!1,raw:t.raw??``,spell:t.spell??!1,phones:t.phones??null}),yt=57600,P=`zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen`.split(` `),bt=`twenty thirty forty fifty sixty seventy eighty ninety`.split(` `),xt=[`thousand`,`million`,`billion`,`trillion`,`quadrillion`,`quintillion`],St={one:`first`,two:`second`,three:`third`,five:`fifth`,eight:`eighth`,nine:`ninth`,twelve:`twelfth`},F=`January February March April May June July August September October November December`.split(` `),I=new Map(F.map((e,t)=>[e.toLowerCase().slice(0,3),t+1])),L=F.join(`|`)+`|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec`,Ct={mon:`Monday`,tue:`Tuesday`,wed:`Wednesday`,thu:`Thursday`,fri:`Friday`,sat:`Saturday`,sun:`Sunday`},R={$:[`dollar`,`dollars`,`cent`,`cents`],"£":[`pound`,`pounds`,`penny`,`pence`],"€":[`euro`,`euros`,`cent`,`cents`],"¥":[`yen`,`yen`,``,``],"₹":[`rupee`,`rupees`,`paisa`,`paise`]};Object.assign(R,{USD:R.$,EUR:R[`€`],GBP:R[`£`],JPY:R[`¥`],INR:R[`₹`],CHF:[`Swiss franc`,`Swiss francs`,`centime`,`centimes`],CAD:[`Canadian dollar`,`Canadian dollars`,`cent`,`cents`],AUD:[`Australian dollar`,`Australian dollars`,`cent`,`cents`],CNY:[`yuan`,`yuan`,``,``]});const wt=`USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY`,Tt={k:`thousand`,m:`million`,b:`billion`,bn:`billion`,t:`trillion`},Et={2:[`half`,`halves`],4:[`quarter`,`quarters`]},Dt={"&":`and`,"@":`at`,"+":`plus`,"=":`equals`,"%":`percent`,"×":`times`,"÷":`divided by`,"±":`plus or minus`,"©":`copyright`,"§":`section`,"°":`degrees`,"≥":`greater than or equal to`,"≤":`less than or equal to`,"≠":`not equal to`,"≈":`approximately`,"∞":`infinity`,"√":`square root of`,π:`pi`},Ot={">=":`greater than or equal to`,"<=":`less than or equal to`,"!=":`not equal to`},kt=new Set(`US IT AI ID UN EU UK LA AD BC AM PM OS PE PA MA LED SAT RIP ER`.split(` `)),At=new Set(`sh shh ss zz hm hmm mm tsk psst pfft hmph psh pssh`.split(` `)),jt=a(String.raw`(?P<gap>[ \t]?)(?P<sign>>=|<=|!=|[<>≥≤≠])(?P=gap)`),Mt=a(String.raw`(?:(?<![\w.,])\d[\d.,]*%?|(?<![\w'’.-])[A-Za-z]|[)\]])$`),Nt=a(String.raw`(?:[-+]?[$£€¥₹]?\d[\d.,]*%?(?!\w)|\.\d|[A-Za-z](?![\w'’])|[(\[])`),Pt=new Set(`if elif while when whenever where whether that because since as and or but so then think thinks thought know knows say says said suppose assume means implies hence thus therefore although though unless until once else also clearly obviously indeed here now again first next finally similarly otherwise however moreover`.split(` `)),Ft=String.raw`[<>+*/^≤≥≠≈×÷±]`,It=String.raw` [<>+*≤≥≠≈×÷±] `,Lt=String.raw`(?![\w'’-]|\s+(?!(?:and|or|then|if|when|where|while|for|is|are|was|but|so|in|to)\b)[A-Za-z])`,Rt=[String.raw`(?<![\w'’.-])(?P<letter>[Aakur])(?=`+``+String.raw`?\s?[<>!=]?=)`,String.raw`[<>!=]?=\s?(?P<letter>[Aakur])`+Lt,String.raw`(?<![\w'’.-])(?P<letter>[Aakur])(?=`+`?`+Ft+String.raw`(?:[\w($]|\.\d))`,String.raw`(?<=[\w)`+`]`+Ft+String.raw`)(?<![Ww]/)(?P<letter>[Aakur])(?![\w'’-])`,String.raw`(?<![\w'’.-])(?P<letter>[Aakur])(?=`+`?`+It+String.raw`(?:[A-Za-z]|\d[\d.,]*)(?![\w'’]))`,String.raw`(?<![\w'’.])(?:[A-Za-z]|\d[\d.,]*)`+`?`+It+`(?P<letter>[Aakur])`+Lt,String.raw`(?<=\()(?P<letter>[Aakur])(?=\)|, )`,String.raw`(?<=, )(?P<letter>[Aakur])(?=\))`,String.raw`(?:(?<=\s)|^)(?P<letter>[akur])(?=\)\s)`].map(e=>a(e)),zt=a(String.raw`(?<![\w)`+``+String.raw`])<[A-Za-z0-9/!?][^<>(){}+*;|\n]*>`+String.raw`|(?<=[\w`+``+String.raw`])<[^<>(){}+*;|\n]{1,80}>(?![\w$])`+String.raw`|</?[A-Za-z][\w-]+/?>`);function Bt(e){let t=e;for(;t.includes(`<`)&&t.includes(`>`);){let e=v(zt,e=>``+e.group(0).slice(1,-1)+``,t);if(e===t)break;t=e}return t}const Vt=a(String.raw`(?<=[\w)`+``+String.raw`])[<>](?=[\w(`+``+String.raw`]|\.\d| minus \d)`),Ht=a(String.raw`(?<![\w.,])\d[\d.,]*%?$`),Ut=a(String.raw`(?:\d[\d.,]*%?(?!\w)|\.\d| minus \d)`),Wt=a(String.raw`(?:(?<![\w'’.])[A-Za-z]|[)`+`])$`),Gt=a(String.raw`(?:[A-Za-z](?![\w'’])|[(`+`])`),z=new Set(`the he she it i we they this that there but and in on at a an if when what why how who my our you then so now after before however`.split(` `)),Kt=new Set(`in since by from until till before after during around circa of year early late mid summer winter spring fall autumn class`.split(` `)),qt=new Set(`war chapter part book volume act scene section phase type class level stage title article appendix world`.split(` `)),Jt=new Set(`plan vitamin grade type option class section part team group exhibit appendix point model series hepatitis level`.split(` `)),Yt={II:2,III:3,IV:4,VI:6,VII:7,VIII:8,IX:9,XI:11,XII:12,XIII:13,XIV:14,XV:15,XVI:16,XVIII:18,XX:20},Xt=/* @__PURE__ */ new Set([`'em`,`'tis`,`'twas`,`'til`,`'cause`,`'bout`,`'round`,`'n`]),Zt=/* @__PURE__ */ RegExp(`[🀀-🫿☀-➿🇦-🇿←-⇿⬀-⯿️‍⃣]+`,`gud`),Qt=a(String.raw`(?<![A-Za-z0-9])(?:[:;=]['^-]?[DPpOo3|/\\*\]\[}{><]+|[)(\]\[][-^']?[:;=]|<3+|</3|\^_?\^|-_-|[oO]_[oO]|>_<|T_T|\\o/|:'[()])(?![A-Za-z0-9])`+String.raw`|[:;]['^-]?[)(]+`+String.raw`|(?<![A-Za-z0-9)])=['^-]?[)(]+|=['^-]?[)(]+(?![A-Za-z0-9(.$-])`+String.raw`|(?<=[A-Za-z]{2})<3+(?![.,]?\d)`),$t=new Set(`call dial ring phone telephone tel mobile cell fax hotline text sms whatsapp`.split(` `)),en=new Set(`to on at or us me him her them back`.split(` `)),tn=new Set(`code pin extension ext passcode password zip postcode cvv cvc otp`.split(` `)),nn=new Set(`number id reference ref serial account tracking booking confirmation`.split(` `)),rn=new Set(`order ticket invoice case claim policy receipt transaction parcel shipment container payment reservation membership customer batch lot item job issue bug request`.split(` `)),an=new Set(`is was are were ends ending in with starts starting begins beginning`.split(` `)),on=new Set(`is was has does goes ships arrives departs leaves shows needs blocks remains includes contains says reads starts ends closes opens runs takes comes this his its us as yes plus minus less`.split(` `)),sn=new Set(`ref reference serial account order invoice tracking part model phone tel fax id room flight case ticket policy licence license booking customer vat tax reg registration`.split(` `)),cn=new Set(`on dated due until till by since from before after for shipped delivered arriving arrives arrive arrived expires expired ends ending starts starting begins beginning monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun`.split(` `)),ln=/* @__PURE__ */ new Set([2,3,4,5,6,8,10,16,32,64,100]),un=new Set(`ratio ratios scale odds aspect mix split proportion`.split(` `)),dn=new Set(`lasted took duration runtime recording video clip track lap pace length long`.split(` `)),fn=new Set(`window slot open opens opening hours shift session served service available booked appointment delivery arrive arrives arriving`.split(` `)),pn=new Set(`IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR BUT NOT ALL`.split(` `)),mn=new Set(`my to in on of go up no do me we us it is at by be so or an as if he hi ok`.split(` `)),hn={".":`dot`,"/":`slash`,"@":`at`,"-":`dash`,_:`underscore`,":":`colon`,"=":`equals`,"&":`and`,"+":`plus`},gn=String.raw`(?:\s?([AaPp])\.[Mm]\.?|\s?([AaPp])[Mm]\b)`,_n=String.raw`(?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<fraction>\d+))?`,vn=a(String.raw`(?:(?<![\w.])(?P<minus>-)\s?)?(?P<symbol>[$£€¥₹]|(?<![A-Za-z])(?:`+wt+String.raw`)(?![A-Za-z]))\s?`+`(?P<inner>-)?`+_n+String.raw`(?:\s?(?P<scale>thousand|million|billion|trillion|bn|[kKmMbB])(?![A-Za-z]))?`),yn=String.raw`[$£€¥₹](?![\w$£€¥₹])|(?:`+`USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY)(?![A-Za-z])`,bn=a(String.raw`(?:(?<![\w.])(?P<minus>-)\s?)?(?<![\w.,^])`+String.raw`(?P<whole>\d{1,3}(?:,\d{3})+|[1-9]\d{0,2}(?: \d{3})+|\d+)(?:\.(?P<fraction>\d+))?`+String.raw`\s?(?P<symbol>`+yn+`)`),xn=a(String.raw`(?<![\w.,-])(\d[\d,]*(?:\.\d+)?)-(?=\d[\d,]*(?:\.\d+)?\s?(?:`+yn+`))`),Sn=a(String.raw`(?<![\w.,])([1-9]\d{0,2}(?:\.\d{3})+),(\d{1,2})(?!\d|[.,]\d)`),Cn=a(String.raw`((?:[$£€¥₹]|(?<![A-Za-z])(?:`+wt+String.raw`))\s?-?\d+),(\d{1,2})(?!\d|[.,]\d)`),wn=a(String.raw`(?<![\w.,])(\d+),(\d{1,2})(?=\s?(?:%|`+yn+`))`),Tn=a(String.raw`(?<![\w$])(?<!\d\s)\$(?:(?=[^\s\d$.,;:!?)])[^$\n]{0,80}`+String.raw`|(?=\d)[^$\n]{0,80}[=^_\\{}+*/<>][^$\n]{0,80}`+String.raw`|(?=\s)[^$\n]{0,80}[=^_\\{}+*<>][^$\n]{0,80})\d\s?\$(?![\w$])`),En=a(String.raw`(?<![\d.,/])(?:(\d+)\s?)?([½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒])`),Dn=a(String.raw`(?P<url>(?:https?://|www\.)[^\s<>\"']+)`+String.raw`|(?P<mail>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)`+String.raw`|(?P<file>(?<![\w@./-])(?P<folder>/?(?:[\w~-]+/)+)?(?P<stem>[\w-]+(?:\.[\w-]+)*)\.(?P<ending>[A-Za-z][A-Za-z0-9]{0,9})`+String.raw`(?![\w-])(?P<path>/[^\s<>\"']*)?)`),On=a(String.raw`\b(`+`Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun|`+L+String.raw`)\b(\.?)\s?[-–]\s?(`+`Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun|`+L+String.raw`)\b(\.?)`),kn=a(String.raw`(?<![\d:.,])(\d{1,3}):(\d{1,3}(?:,\d{3})+|\d+)(?![\d:]|[.,]\d)`),An=a(String.raw`(?<![\w.,/-])(#)?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![\w-]|[.,]\d)`),jn=a(String.raw`(?<![\w.,/-])(\d+(?:/\d+){2,})(?![\w/]|[.,]\d)`),Mn=String.raw`[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+`,Nn=a(String.raw`(?<![\w.,:/#+-])(?P<plus>\+)?`+String.raw`(?:(?P<lead>[A-Z]{2}) (?P<core>`+Mn+String.raw`)(?: (?P<trail>[A-Z]{2})(?![\w]))?`+String.raw`|(?P<core2>`+Mn+`))`+String.raw`(?![\w]|[.,:/-]\d|\s?[%°])`),Pn=a(String.raw`(?<![\w'.])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?![\w']|\.\d)`),Fn={résumé:`R EH1 Z AH0 M EY2`,resumé:`R EH1 Z AH0 M EY2`,exposé:`EH2 K S P OW0 Z EY1`,rosé:`R OW0 Z EY1`,pâté:`P AA0 T EY1`,maté:`M AA1 T EY2`,saké:`S AA1 K EY2`,lamé:`L AA0 M EY1`},In=a(String.raw`\b(`+Object.keys(Fn).join(`|`)+String.raw`)(s?)\b`,`i`),Ln={ß:`ss`,æ:`ae`,Æ:`Ae`,œ:`oe`,Œ:`Oe`,ø:`o`,Ø:`O`,đ:`d`,ł:`l`,Ł:`L`};function Rn(e){if(c(e)||!n(e))return e;if(e in Ln)return Ln[e];let t=[...e.normalize(`NFKD`)].filter(e=>!/\p{M}/u.test(e)).join(``);return c(t)&&n(t)?t:e}function B(e){let t=0n;for(let n of e){let e=n.codePointAt(0);if(e>=48&&e<=57){t=t*10n+BigInt(e-48);continue}let r=e;for(;/\p{Nd}/u.test(String.fromCodePoint(r-1))&&e-r<9;)r--;e-=r,t=t*10n+BigInt(e)}return t}const V=e=>Number(B(e));function H(e){let t=BigInt(e);if(t<0n)return`minus `+H(-t);if(t<20n)return P[Number(t)];if(t<100n){let e=Number(t/10n),n=Number(t%10n);return bt[e-2]+(n?` `+P[n]:``)}if(t<1000n){let e=Number(t/100n),n=t%100n;return P[e]+` hundred`+(n?` `+H(n):``)}if(t>=1000n**BigInt(xt.length+1))return W(t.toString());let n=[],r=t;for(let e=xt.length;e>0;e--){let t=1000n**BigInt(e),i=r/t;r%=t,i&&n.push(H(i)+` `+xt[e-1])}return r&&n.push(H(r)),n.join(` `)}function U(e){let t=H(e).split(` `),n=t[t.length-1];return t[t.length-1]=St[n]??(n.endsWith(`y`)?n.slice(0,-1)+`ieth`:n+`th`),t.join(` `)}function W(e,t=`zero`){return[...e].filter(e=>g(e)).map(e=>e===`0`?t:P[V(e)]).join(` `)}function G(e){if(!(e>=1e3&&e<=2999))return H(e);let t=Math.floor(e/100),n=e%100;return n===0?t%10==0?H(e):H(t)+` hundred`:t===20&&n<10?`two thousand `+P[n]:H(t)+(n<10?` oh `+P[n]:` `+H(n))}function zn(e){let t=Math.floor(e/100),n=e%100;return!(e>=100&&e<=9999)||n===0&&(t<10||t%10==0)?H(e):n===0?H(t)+` hundred`:H(t)+(n<10?` oh `+P[n]:` `+H(n))}function K(e){let t=e.replaceAll(`,`,``),n=t.indexOf(`.`),r=n<0?t:t.slice(0,n),i=n<0?``:t.slice(n+1);return i?(r?H(B(r)):``)+` point `+W(i):r.length>1&&r.startsWith(`0`)||r.length>18?W(r):H(B(r))}function Bn(e,t,n){let r=[[e,`hour`],[t,`minute`],[n,`second`]].filter(([e])=>e).map(([e,t])=>`${H(e)} ${e===1?t:t+`s`}`);return r.length<2?r.length?r[0]:`zero seconds`:r.slice(0,-1).join(`, `)+` and `+r[r.length-1]}function Vn(e){let t=e.lastIndexOf(` `),n=t<0?``:e.slice(0,t),r=t<0?e:e.slice(t+1);return r=r.endsWith(`y`)?r.slice(0,-1)+`ies`:r+`s`,(n?n+` `:``)+r}function Hn(e,t,n=!1){let[r,i]=Et[t]??[U(t),U(t)+`s`];return e===1?(n?`aeiou`.includes(r[0])?`an `:`a `:`one `)+r:`${H(e)} ${i}`}const Un=e=>e.length>2?W(e):K(e),Wn=e=>e.startsWith(`0`)?W(e,`oh`):H(B(e));function Gn(e){if(e.startsWith(`0`))return!1;let t=new Set(e.slice(1));return e.length<=2||t.size===1&&t.has(`0`)}const q=e=>`${e}`,Kn=/[.!?;…]\s|[()"]/u;function J(e,t,n=4){let r=e.slice(Math.max(0,t-80),t).split(Kn);return(r[r.length-1].match(/[A-Za-z]+/g)??[]).map(e=>e.toLowerCase()).slice(-n)}function qn(e,t){let n=J(e,t);if(!n.length)return null;let r=n[n.length-1];if($t.has(r)||en.has(r)&&n.slice(-3,-1).some(e=>$t.has(e)))return`phone`;let i=n.length-1;for(;i>0&&an.has(n[i]);)i--;for(let[e,t]of[[`phone`,$t],[`pin`,tn],[`strong`,nn],[`weak`,rn]])if(t.has(n[i]))return e;return r===`enter`?`enter`:null}const Jn=a(String.raw`\d+,\d{1,2}`),Yn=e=>d(Jn,e)?e.replaceAll(`,`,`.`):e;function Xn(e){return e.length<=2&&(e.length!==2||e[0]!==`0`)?H(B(e)):e.length===4&&V(e)>=1900&&V(e)<=2099?G(V(e)):W(e)}const Zn=(e,t)=>e!==void 0&&t.includes(e),Y=a(String.raw`\s*([A-Za-z]+)`),Qn=a(String.raw`[A-Za-z']+`);function $n(e){return[...e].map(e=>{let t=e.toLowerCase(),n=e.toUpperCase();return t!==n&&[...t].length===1&&[...n].length===1?`[${t}${n}]`:u(e)}).join(``)}var er=class l{abbreviations=/* @__PURE__ */ new Map;units=/* @__PURE__ */ new Map;exactUnits=/* @__PURE__ */ new Map;chat=/* @__PURE__ */ new Map;replies=/* @__PURE__ */ new Set;endings;unitNames;unit;area=a(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft|mi|yd|in)([²³])`);areaPlain=a(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft)([23])(?![\w])`);inline=[];dayFirst=!1;dated=!1;isWord;constructor(e=()=>!0){this.isWord=e;for(let[,e]of D(`abbreviations.txt`))this.abbreviations.set(e[0],[e[1],e[2]]);for(let[,e]of D(`units.txt`))e.length>3&&e[3]===`exact`?this.exactUnits.set(e[0],[e[1],e[2]]):this.units.set(e[0].toLowerCase(),[e[1],e[2]]);for(let[e,t]of D(`chat.txt`)){if(t.length>2&&(t.length!==3||t[2]!==`alone`))throw Error(`chat.txt line ${e}: after the words to say there can only be "alone"`);this.chat.set(t[0],t.length>1?t[1]:`-`),t.length>2&&this.replies.add(t[0])}this.endings=We(`endings.txt`);let t=[.../* @__PURE__ */ new Set([...this.units.keys(),...this.exactUnits.keys()])].sort((e,t)=>[...t].length-[...e].length||(e<t?-1:+(e>t)));this.unitNames=t.map(u).join(`|`),this.unit=a(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[-–](\d[\d,]*(?:\.\d+)?))?\s?(`+t.map($n).join(`|`)+String.raw`)(?![A-Za-z])(\.(?=\s+[a-z]))?`)}amount(e,t,n,r){let[i,a,o,s]=R[e];if(r){let e=Tt[r.toLowerCase()]??r.toLowerCase();return`${K(t+(n?`.`+n:``))} ${e} ${a}`}if(n&&(n.length>2||!o))return`${K(t+`.`+n)} ${a}`;let c=B(t),l=V((n||`0`).padEnd(2,`0`)),u=[];return(c||!l)&&u.push(`${H(c)} ${c===1n?i:a}`),l&&o&&u.push(`${H(l)} ${l===1?o:s}`),u.join(` and `)}static continental(e){let t=e.group(1).split(`.`);return t.length===4&&t.every(e=>V(e)<=255)?e.group(0):t.join(`,`)+`.`+e.group(2)}money(e){let t=e.group(`whole`).replaceAll(`,`,``).replaceAll(` `,``),n=this.amount(e.group(`symbol`),t,e.group(`fraction`),e.group(`scale`));return(e.group(`minus`)||e.group(`inner`)?` minus `:` `)+n+` `}static halfDay(e,t,n){if(!(t||n))return``;let r=e.string.slice(e.end()).trimStart(),i=o(Qn,r),a=!!t&&e.group(0).endsWith(`.`)&&(!r||f(r[0])&&i!==null&&z.has(i.group(0).toLowerCase()));return`${(t||n).toUpperCase()}M${a?` .`:``} `}time(e){let t=V(e.group(1)),n=V(e.group(2)),r=e.group(3),i=e.group(4)||e.group(5),a=J(e.string,e.start(),3);if(r!==void 0&&!i&&!(a.length&&a[a.length-1]===`at`))return` ${Bn(t,n,V(r))} `;if(t>24&&r===void 0&&!i){let r=o(Y,e.string.slice(e.end()));return a.some(e=>dn.has(e))||r&&dn.has(r.group(1).toLowerCase())?` ${Bn(0,t,n)} `:` ${H(t)} ${n<10?`oh `:``}${H(n)} `}if(t>24)return e.group(0);let s=H(t);return n===0?i||(t===0?s=`midnight`:s+=t>12?` hundred`:` o'clock`):s+=(n<10?` oh `:` `)+H(n),r&&(s+=` and `+H(V(r))+(V(r)===1?` second`:` seconds`)),` ${s} `+l.halfDay(e,e.group(4),e.group(5))}dotTimes(e){let t=e.string.slice(e.end());return V(e.group(1))>23||V(e.group(4))>23||!J(e.string,e.start()).some(e=>fn.has(e))||o(a(String.raw`\s?(?:%|percent|million|billion|`+this.unitNames+String.raw`)(?![A-Za-z])`,`i`),t)?e.group(0):`${e.group(1)}:${e.group(2)}${e.group(3)}${e.group(4)}:${e.group(5)}`}ratio(e){let t=e.group(1),n=e.group(2);if(t.length<=2&&n.length===2&&`012345`.includes(n[0])){let t=o(Y,e.string.slice(e.end()));if(!(J(e.string,e.start(),3).some(e=>un.has(e))||t&&un.has(t.group(1).toLowerCase())))return e.group(0)}return` ${K(t)} to ${K(n)} `}date(e,t,n,r=!1){if(!(e>=1&&e<=12)||!(t>=1&&t<=31))return null;let i=r?`the ${U(t)} of ${F[e-1]}`:`${F[e-1]} ${U(t)}`;return n&&(i+=` `+(n.length===4?G(V(n)):(V(n)<10?`oh `:``)+H(B(n)))),` ${i} `}slashDate(e){let t=V(e.group(1)),n=V(e.group(2));return(t>12&&12>=n||this.dayFirst&&n<=12)&&([t,n]=[n,t]),this.date(t,n,e.group(3))??e.group(0)}dotDate(e){let t=V(e.group(1)),n=V(e.group(2));return n>12&&12>=t&&([t,n]=[n,t]),this.date(n,t,e.group(3))??e.group(0)}dayMonth(e){let t=e.group(2),n=this.date(I.get(t.toLowerCase().slice(0,3)),V(e.group(1)),e.group(4),!0);return n===null?e.group(0):n+(e.group(3)&&F.includes(t)?`. `:``)}daysOfMonth(e){let t=n(e.group(1)[0]),[r,i,a]=t?[e.group(1),e.group(2),e.group(3)]:[e.group(3),e.group(1),e.group(2)],o=V(i),s=V(a);if(!(o>=1&&o<=31&&s>=1&&s<=31))return e.group(0);let c=F[I.get(r.toLowerCase().slice(0,3))-1];return` ${t?`${c} ${U(o)} to ${U(s)}`:`the ${U(o)} to the ${U(s)} of ${c}`}${e.group(4)?` `+G(V(e.group(4))):``} `}dayRange(e){let t=[];for(let n of[e.group(1),e.group(3)]){let e=n.toLowerCase();t.push(this.abbreviations.has(e)?this.abbreviations.get(e)[1]:Ct[e]??n)}let n=e.string.slice(e.end()).trimStart(),r=o(Qn,n),i=!!e.group(4)&&(!n||f(n[0])&&r!==null&&z.has(r.group(0).toLowerCase()));return`${t[0]} to ${t[1]}${i?` .`:``}`}shortDate(e){let t=e.group(1),n=e.group(2);if(t.length>2||n.length>2)return null;let r=V(t),i=V(n),a=J(e.string,e.start());if(r>=1&&r<=12&&i>=20&&i<=49&&a.includes(`card`))return` ${F[r-1]} ${G(2e3+i)} `;let o=a.length?a[a.length-1]:``,s=r<i&&ln.has(i);return t.length===2&&t[0]===`0`||n.length===2&&n[0]===`0`||r>12&&r<=31&&i<=12&&(r!==24||i!==7)||cn.has(o)&&(!s||o===`on`)||this.dated&&!s&&r!==i?((r>12||this.dayFirst&&i<=12)&&([r,i]=[i,r]),this.date(r,i)):null}fraction(e){let t=this.shortDate(e);if(t)return t;let n=B(e.group(1)),r=B(e.group(2));return n===24n&&r===7n||n===r?` ${H(n)} ${H(r)} `:!(n>0n&&n<100n)||!(r>1n&&r<=100n)?` ${H(n)} over ${H(r)} `:` ${Hn(Number(n),Number(r))} `}mixedFraction(e){let t=V(e.group(2)),n=V(e.group(3));return t>0&&t<n?` ${K(e.group(1))} and ${Hn(t,n,!0)} `:e.group(0)}vulgar(e){let[t,n]=e.group(2).normalize(`NFKC`).split(`⁄`).map(V);return e.group(1)===void 0?` ${t}/${n} `:` ${H(B(e.group(1)))} and ${Hn(t,n,!0)} `}static phone(e){return` `+e.groups().filter(e=>!!e).map(e=>e===`800`?`eight hundred`:W(e,`oh`)).join(`, `)+` `}measure(e){let t=e.group(3),n=this.exactUnits.get(t)??this.units.get(t.toLowerCase());if(n===void 0||[...t].length===1&&[`-`,`'`,`/`].includes(e.string.slice(e.end(3),e.end(3)+1)))return e.group(0);let[r,i]=n,a=Yn(e.group(1)),o=e.group(2)&&Yn(e.group(2));return o?` ${K(a)} to ${K(o)} ${i} `:` ${K(a)} ${a===`1`||a===`1.0`?r:i} `}square(e){let t=e.group(2),[n,r]=t===`in`?[`inch`,`inches`]:this.exactUnits.get(t)??this.units.get(t),i=`²2`.includes(e.group(3))?`square`:`cubic`,a=Yn(e.group(1));return` ${K(a)} ${i} ${a===`1`||a===`1.0`?n:r} `}counted(e,t){let n=o(Y,e.slice(t));if(!n)return!1;let r=n.group(1).toLowerCase();return[`people`,`men`,`women`,`children`,`feet`,`miles`].includes(r)?!0:r.length>3&&r.endsWith(`s`)&&!r.endsWith(`ss`)&&!on.has(r)&&this.isWord(r.slice(0,-1))}number(e,t){let n=e.group(0),r=n.replaceAll(`,`,``);if(!r.includes(`.`)&&!n.includes(`,`)&&!r.startsWith(`0`)&&r.length>2){let n=qn(t,e.start()),i={phone:3,pin:3,strong:4,enter:4,weak:5}[n??``]??7;if(r.length>=i&&!(n===null&&r.endsWith(`000`))&&!(n===`enter`&&r.length===4&&V(r)>=1900&&V(r)<=2099)&&(n!==null&&n!==`phone`&&n!==`weak`||!this.counted(t,e.end())))return` ${W(r,n===`phone`?`oh`:`zero`)} `}if(!r.includes(`.`)&&!n.includes(`,`)&&r.length===4&&!r.startsWith(`0`)){let n=V(r),i=t.slice(0,e.start()).slice(-24).match(/[A-Za-z]+/g)??[],s=o(Y,t.slice(e.end())),c=!!s&&/(?:[^s]s|people|men|women|children|feet|miles)$/.test(s.group(1).toLowerCase()),l=i.length?i[i.length-1].toLowerCase():``,u=i.length>0&&(Kt.has(l)||I.has(l.slice(0,3))),d=_(a(String.raw`(?<![A-Za-z])([A-Z]{2,5}) $`),t.slice(0,e.start()));if(d&&!pn.has(d.group(1))&&!c)return` ${zn(n)} `;if(n>=1100&&n<=1999||n>=2e3&&n<=2099&&(u||!c))return` ${G(n)} `}return` ${K(n)} `}readsAsWord(e){let t=e.toLowerCase();return t.length<3||!this.isWord(t)?!1:t.length>3||[`s`,`es`,`ed`,`ing`].some(e=>this.isWord(t+e))}static alnum(e,t=`zero`){return p(a(`[A-Za-z]+|\\d+`),e).map(e=>g(e)?W(e,t):q(e)).join(` `)}static digitGroups(e,t=`zero`){return e.every(e=>Gn(e)&&e.length<=2)?e.map(e=>H(B(e))).join(` , `):e.map(e=>W(e,t)).join(` , `)}codeWords(e,t){let n=e.join(`-`);if(![...n].some(e=>g(e)))return null;let r=e.map(e=>g(e)?`d`:[...e].some(e=>g(e))?`m`:f(e)?`u`:`l`);if(r.every(e=>e===`d`)){if(e.length===2){let n=e.every(e=>e.length===4&&V(e)>=1e3&&V(e)<=2100);if(!(t.group(1)||e.some(e=>e.length>1&&e[0]===`0`)||Math.min(...e.map(e=>e.length))>=4&&!n||[`strong`,`pin`].includes(qn(t.string,t.start())??``)))return null}return l.digitGroups(e)}if(e.length===2&&r[1]===`d`&&`ul`.includes(r[0])){let[t,n]=e;return r[0]===`l`&&t.length>1||this.readsAsWord(t)?null:Gn(n)?`${q(t)} ${H(B(n))}`:`${q(t)} , ${W(n)}`}if(e.length===2&&r[0]===`d`&&(r[1]===`l`||[...e[1]].length===1||d(a(String.raw`\d+[A-Za-z]+`),e[1]))||r.some((t,n)=>t===`l`&&e[n].length>2||t===`m`&&d(a(String.raw`\d+(?:s|st|nd|rd|th)`),e[n]))||![...n].some(e=>f(e))&&(r.includes(`l`)||r.includes(`m`)))return null;let i=[];return e.forEach((e,t)=>{let n=r[t];n===`d`?i.push(W(e)):n===`m`?i.push(l.alnum(e)):e.length>3&&this.isWord(e.toLowerCase())?i.push(e):i.push(e.length<=5?q(e):e)}),i.join(` , `)}code(e){let t=this.codeWords(e.group(2).split(`-`),e);return t===null?e.group(0):(e.group(1)?` number `:` `)+t+` `}grouped(e){let t=e.string,n=e.group(`plus`),r=e.group(`lead`),i=e.group(`trail`),s=(e.group(`core`)??e.group(`core2`)).split(` `),c=``,u=``;r!==void 0&&pn.has(r)&&(c=r+` `,u=i?` `+i:``,r=void 0,i=void 0);let d=qn(t,e.start()),f=s.every(e=>g(e));if(f&&!n&&!r){let n=t.slice(e.end()),r=this.counted(t,e.end())||o(a(String.raw`\s?(?:`+this.unitNames+String.raw`)(?![A-Za-z])`,`i`),n)!==null;if(d===null&&s[0][0]!==`0`&&s[0].length<=3&&s.slice(1).every(e=>e.length===3)&&(s[s.length-1]===`000`||r&&s.length>2))return c+s.join(`,`)+u;if(!(d||s[0].length>1&&s[0][0]===`0`||s.length>2&&s.some(e=>e.length>2)))return e.group(0)}else if(!f&&!r&&!n&&d===null)return e.group(0);let p=n||d===`phone`||f&&s[0][0]===`0`&&s.reduce((e,t)=>e+t.length,0)>=10?`oh`:`zero`,m=f?l.digitGroups(s,p):s.map(e=>g(e)?W(e,p):l.alnum(e,p)).join(` , `);return r&&(m=`${q(r)} , ${m}`),i&&(m=`${m} , ${q(i)}`),c+(n?` plus `:` `)+m+` `+u}glued(e){let t=[];for(let n of p(a(`[A-Za-z]+|\\d+`),e.group(0)))g(n)?t.push(Gn(n)?H(B(n)):W(n)):n.length===1||(f(n)||n.length<=3)&&!this.readsAsWord(n)?t.push(q(n)):t.push(n);return` `+t.join(` `)+` `}addressPart(e){let t=e.toLowerCase();return e.length>1&&this.isWord(t)&&(e.length>2||mn.has(t))?e:e.length<=3?q(e):e}spokenAddress(e){let t=p(a(`[A-Za-z]+|\\d+|.`),e),r=e.includes(`@`),i=!1,o=[];return t.forEach((e,s)=>{let c=s+1<t.length?t[s+1]:``;if(e===`@`&&(i=!0),g(e))o.push(Xn(e));else if(!n(e))o.push(hn[e]??``);else if(s&&t[s-1]===`.`&&(i||!r)&&([``,`/`,`:`,`?`,`#`].includes(c)||e.length<=3&&(this.endings.has(e.toLowerCase())||!this.isWord(e.toLowerCase()))))o.push(this.endings.get(e.toLowerCase())===`word`?e.toLowerCase():q(e));else{let t=``;for(let n of p(a(`[A-Z]?[a-z]+|[A-Z]+(?![a-z])`),e)){let e=this.addressPart(n);e.startsWith(``)?t+=n:(t&&=(o.push(q(t)),``),o.push(e))}t&&o.push(q(t))}}),o.filter(Boolean).join(` `)}address(t){let n=t.group(0),i=e(n,`.,;:!?)`),a=n.slice(i.length);if(t.group(`file`)){let e=t.group(`ending`),i=this.endings.has(e.toLowerCase()),a=!!(t.group(`folder`)||t.group(`path`))||t.group(`stem`).includes(`_`);if(!(i||a&&e.length<=5&&(r(e)||f(e))&&!this.isWord(e.toLowerCase())))return n}let o=e(i.replace(/^(?:https?:\/\/)?(?:www\.)?/,``),`/`);return` ${this.spokenAddress(o)} ${a}`}chatForm(e){let t=this.chat.get(e.group(0).toLowerCase());return t===void 0?e.group(0):t===`-`?` `:` `+y(t).map(e=>e.startsWith(`*`)?`${e.slice(1)}`:e).join(` `)+` `}inlined(e){let t=y(e.group(2));if(!t.length||t.every(e=>[...e].length===1&&c(e)))return e.group(0);let n;try{n=Se(t)?Me(``,t):E(t.join(` `))}catch{return e.group(0)}return this.inline.push([e.group(1)||``,n]),` ${String.fromCodePoint(yt+this.inline.length-1)} `}static dotted(e,t){let n=e.group(0).replaceAll(`.`,``),r=n.toLowerCase(),i=t.slice(e.end()).trimStart(),a=r===`eg`?`for example`:r===`ie`?`that is`:`${n.toUpperCase()}`,s=o(Qn,i);return` ${a}${(!i||f(i[0])&&s&&z.has(s.group(0).toLowerCase()))&&e.group(0).endsWith(`.`)?` .`:``} `}abbreviation(e,t){let n=e.group(1),i=e.group(2),s=n.toLowerCase(),c=t.slice(e.end()),l=o(a(String.raw`\s*([A-Za-z]+|\d)`),c),u=!!l&&f(l.group(1)[0]),d=_(a(String.raw`[A-Z][a-z]+\s+$`),t.slice(0,e.start()))!==null;if((s===`st`||s===`dr`)&&f(n[0]))return u&&(i||s===`dr`)?s===`st`?` saint `:` doctor `:d&&i?(s===`st`?` street`:` drive`)+(c.trim()?` `:` .`):e.group(0);if(!this.abbreviations.has(s))return e.group(0);let[p,m]=this.abbreviations.get(s),h=m,v=!!l&&g(l.group(1));if(p===`title`){if(!(f(n[0])&&(u||i)))return e.group(0)}else if(p===`number`){let n=_(a(String.raw`([A-Za-z]+)\.?\s+$`),t.slice(0,e.start()));if(!(i&&(v||s===`no`&&n&&sn.has(n.group(1).toLowerCase()))))return e.group(0)}else if(!i||p===`small`&&f(n))return e.group(0);else if([`mar`,`sun`,`sat`,`jan`,`dec`,`max`,`min`,`est`,`no`].includes(s)&&!f(n[0])&&!v&&!([`max`,`min`,`est`].includes(s)&&l&&r(l.group(1)[0])))return e.group(0);s===`est`&&!v&&(h=`estimated`);let y=p!==`title`&&(!c.trim()||u&&z.has(l.group(1).toLowerCase()));return` ${h}${y?` .`:``} `}rewrite(e){let t=v(En,e=>this.vulgar(e),e);t=v(this.area,e=>this.square(e),t),t=t.replaceAll(`²`,` squared `).replaceAll(`³`,` cubed `),t=t.normalize(`NFKC`).replaceAll(`⁄`,`/`).replaceAll(`−`,` minus `),c(t)||(t=v(In,e=>`{`+Fn[e.group(1).toLowerCase()]+(e.group(2)?` Z`:``)+`}`,t),t=[...t].map(Rn).join(``)),t=v(a(`(?<=\\d)–(?=\\d)`),`-`,t),t=v(On,e=>this.dayRange(e),t);for(let[e,n]of[[`‘`,`'`],[`’`,`'`],[`‚`,`'`],[`‛`,`'`],["`",`'`],[`´`,`'`],[`“`,`"`],[`”`,`"`],[`„`,`"`],[`‟`,`"`],[`«`,`"`],[`»`,`"`],[`–`,`—`],[`―`,`—`],[`[`,`(`],[`]`,`)`]])t=t.replaceAll(e,n);return this.inline=[],t=v(a(String.raw`([A-Za-z']+)?\{([^{}]*)\}`),e=>this.inlined(e),t),t=t.replaceAll(`{`,`(`).replaceAll(`}`,`)`),t=v(Dn,e=>this.address(e),t),t=t.replaceAll(`→`,` to `),t=v(Zt,` `,t),t=v(Qt,` `,t),t=v(a(String.raw`\.{3,}`),`…`,t),t=v(a(String.raw`(?<=\d)\s-\s(?=\d[\d.,]*\s?=)`),` minus `,t),t=v(a(String.raw`(?<![\d.,/-])(\d[\d.,]*)-(?=\d[\d.,]*=)`),`\\1 minus `,t),t=v(a(String.raw`-{2,}|\s-\s`),`—`,t),t=v(a(String.raw`\bw/o\b`,`i`),` without `,t),t=v(a(String.raw`\bw/(?=\s|$)`,`i`),` with `,t),t=v(a(String.raw`\bn/a\b`,`i`),` not applicable `,t),t=t.replaceAll(`'n'`,` 'n `),_(a(`[<>≥≤≠]|!=`),t)&&(t=l.markComparisons(t)),_(a(`\\d`),t)&&(t=this.figures(t)),this.signs(t)}figures(e){let t=v(Tn,e=>e.group(0).slice(0,-1)+``,e);t=v(Sn,e=>l.continental(e),t),t=v(Cn,`\\1.\\2`,t),t=v(wn,`\\1.\\2`,t),t=v(vn,e=>this.money(e),t),t=v(xn,`\\1 to `,t),t=v(bn,e=>this.money(e),t),t=v(a(String.raw`(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?¢`),e=>` ${K(e.group(1))} ${e.group(1)===`1`?`cent`:`cents`} `,t);let n=String.raw`\d{1,2}:[0-5]\d(?:\s?[AaPp]\.?[Mm]\.?)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\.?`;t=v(a(`(`+n+String.raw`)\s(`+`ET|CT|MT|PT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC|BST|CET|CEST|IST|AEST|AEDT|JST`+String.raw`)\b`),e=>`${e.group(1)} ${q(e.group(2))}`,t),t=v(a(String.raw`(?<![\d.,:])(\d{1,2})\.([0-5]\d)(?=\s?[AaPp]\.?[Mm]\b)`),`\\1:\\2`,t),t=v(a(String.raw`(?<![\d.,:])(\d{1,2})\.(00|15|30|45)(\s?(?:to|until|till|and|[-—])\s?)(\d{1,2})\.(00|15|30|45)(?!\d|\.\d)`),e=>this.dotTimes(e),t),t=v(a(`(`+n+String.raw`)\s?[-—]\s?(?=\d{1,2}(?::[0-5]\d|\s?[AaPp]\.?[Mm]\b))`),`\\1 to `,t),t=v(kn,e=>this.ratio(e),t),t=v(a(String.raw`(?<![\d:])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?`+gn+String.raw`?(?![\d:])`),e=>this.time(e),t),t=v(a(String.raw`(?<![\d.:,])(\d{1,2})`+gn+String.raw`(?![A-Za-z])`),e=>` ${H(B(e.group(1)))} `+l.halfDay(e,e.group(2),e.group(3)),t);let r=s(a(String.raw`(?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(?:\d{4}|\d{2}))?(?![\d/])`),t).map(([e,t])=>[V(e),V(t)]).filter(([e,t])=>e!==24||t!==7);return this.dayFirst=r.some(([e,t])=>e>12&&12>=t)&&!r.some(([e,t])=>t>12&&12>=e),this.dated=_(a(String.raw`(?<![\d/])\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?![\d/])`),t)!==null,t=v(a(String.raw`(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])`),e=>this.date(V(e.group(2)),V(e.group(3)),e.group(1))??e.group(0),t),t=v(a(String.raw`(?<![\d/])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])`),e=>this.date(V(e.group(2)),V(e.group(3)),e.group(1))??e.group(0),t),t=v(a(String.raw`(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d|\.\d)`),e=>this.dotDate(e),t),t=v(a(String.raw`(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])`),e=>this.slashDate(e),t),t=v(a(String.raw`\b(`+L+String.raw`)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])`),e=>this.daysOfMonth(e),t),t=v(a(String.raw`(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?\s+(`+L+String.raw`)\b(?:,?\s+(\d{4}))?`),e=>this.daysOfMonth(e),t),t=v(a(String.raw`\b(`+L+String.raw`)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:])`),e=>this.date(I.get(e.group(1).toLowerCase().slice(0,3)),V(e.group(2)),e.group(3))??e.group(0),t),t=v(a(String.raw`(?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s+(`+L+String.raw`)\b(\.(?=\s+[a-z\d]))?(?:,?\s+(\d{4}))?`),e=>this.dayMonth(e),t),t=v(a(String.raw`(?<![\d-])(?:\+?(1)[-.\s])?\(?(\d{3})\)?[-.\s](\d{3})[-.\s](\d{4})(?![\d-])`),l.phone,t),t=v(a(String.raw`(?<![\w.,-])(\d{3})-(\d{4})(?![\w-])`),l.phone,t),t=v(a(String.raw`(?<![\w.])([vV])?(\d+(?:\.\d+){2,})(?:-(\d+)(?![\w-]|\.\d))?(?![\w])`),e=>` `+(e.group(1)?q(e.group(1))+` `:``)+e.group(2).split(`.`).map(Un).join(` point `)+(e.group(3)?` dash `+K(e.group(3)):``)+` `,t),t=v(a(String.raw`(?<![\w.])([vV])(\d+)\.(\d{2,3})(?!\d|\.\d|\w)`),e=>` ${q(e.group(1))} ${H(B(e.group(2)))} point ${Wn(e.group(3))} `,t),t=v(a(String.raw`\b(`+`version|ver|release|firmware|patch|kernel|build|python|node|nodejs|ios|ipados|macos|watchos|tvos|android|chrome|firefox|safari|java|php|ruby|rust|swift|kotlin|golang|django|rails|react|angular|vue|docker|kubernetes|ubuntu|debian|fedora|windows|linux|postgres|postgresql|mysql|redis|numpy|pytorch|tensorflow|cuda|xcode|gcc|clang|unity|minecraft|typescript|bluetooth|usb|hdmi|opengl|vulkan|directx|upgrade[ds]?|update[ds]?|downgrade[ds]?`+String.raw`)(\.?\s(?:(?:from|to)\s)?)(\d+)\.(\d{2,3})(?:(\s(?:to|or|and)\s)(\d+)\.(\d{2,3}))?(?!\d|\.\d|\s?%|\s(?:percent|million|billion)\b)`,`i`),e=>`${e.group(1)}${e.group(2)}${H(B(e.group(3)))} point ${Wn(e.group(4))}`+(e.group(5)?`${e.group(5)}${H(B(e.group(6)))} point ${Wn(e.group(7))}`:``)+` `,t),t=v(An,e=>this.code(e),t),t=v(jn,e=>` `+l.digitGroups(e.group(1).split(`/`))+` `,t),t=v(Nn,e=>this.grouped(e),t),t=v(a(String.raw`(?<![\w.])-(?=\d)`),` minus `,t),t=v(a(String.raw`(?:(?<=\d)\s?[xX×]|×)\s?(?=\d[\d.,]*\s?(?:`+`mm|cm|km|m|ft|in|inch|inches|feet|foot|metres?|meters?|yards?|yd|px|pixels?`+String.raw`)\b)`),` by `,t),t=v(a(String.raw`(?<=\d)\s[xX×]\s(?=\d)`),` times `,t),t=v(a(String.raw`(?<![\w/'])[A-Za-z0-9/]*[0-9/][A-Za-z0-9/]*(?![\w/'])`),e=>this.chatForm(e),t),t=v(a(String.raw`(?<![\w.])'?(\d{2}|\d{4})'?s\b`),e=>` `+Vn(e.group(1).length===4?G(V(e.group(1))):H(B(e.group(1))))+` `,t),t=v(a(String.raw`(?<![\w.])(\d+)(?:st|nd|rd|th)\b`),e=>` ${U(B(e.group(1)))} `,t),t=v(a(String.raw`(?<![\w.])(144|240|360|480|540|576|720|900|1080|1440|2160|4320)([pi])\b`),e=>` ${zn(V(e.group(1)))} ${q(e.group(2))} `,t),t=v(a(String.raw`(\d[\d,]*(?:\.\d+)?)\s?%`),e=>` ${K(e.group(1))} percent `,t),t=v(this.areaPlain,e=>this.square(e),t),t=v(this.unit,e=>this.measure(e),t),t=v(a(String.raw`(?<![\d/.,-])(\d+) (\d)/(2|3|4|5|6|8|10|16)(?![\d/]|\.\d)`),e=>this.mixedFraction(e),t),t=v(a(String.raw`(?<![\d/.])(\d+)/(\d+)(?![\d/]|\.\d)`),e=>this.fraction(e),t),t=v(a(String.raw`(?<![\w.,-])(\d{4})[-–](\d{4})(?![\w.,-])`),e=>` ${G(V(e.group(1)))} to ${G(V(e.group(2)))} `,t),t=v(a(String.raw`(?<![\w.,-])(\d+)[-–](\d+)(?![\w-]|[.,]\d)`),e=>` ${K(e.group(1))} to ${K(e.group(2))} `,t),t=v(a(String.raw`(?<![\w.])(\d+)x\b`),e=>` ${K(e.group(1))} times `,t),t=v(a(String.raw`(?<=\d)\^(?=\d)`),` to the power of `,t),t}static namedLetters(e){if(![...`=<>+*/^()≤≥≠≈×÷±`].some(t=>e.includes(t)))return e;let n=Bt(e),r=/* @__PURE__ */ new Set;for(let e of Rt)for(let i of t(e,n))r.add(i.start(`letter`));if(!r.size)return e;let i=``;for(let t=0;t<e.length;t++)i+=r.has(t)?`${e[t].toUpperCase()}`:e[t];return i}static markComparisons(e){let t=Bt(e);return v(jt,e=>{let n=t.slice(0,e.start()),r=t.slice(e.end()),i=_(Mt,n);if(t[e.start(`sign`)]!==e.group(`sign`)[0]||i===null||!o(Nt,r))return e.group(0);let s=i.start();if(`)]`.includes(i.group(0))&&i.group(0).length===1){let e=0;for(;s>=0&&(e+=+!!Zn(n[s],`)]`)-!!Zn(n[s],`([`),e!==0);)--s;for(;s>0&&(m(n[s-1])||`_.`.includes(n[s-1]));)--s}let c=_(a(String.raw`[A-Za-z']+$`),n.slice(0,Math.max(s,0)).trimEnd());return c&&!Pt.has(c.group(0).toLowerCase())?e.group(0):``+e.group(0)},e)}static compared(t,n){return` ${t.group(0).startsWith(``)||e(t.string.slice(0,t.start()),` `).endsWith(``)?`is `:``}${n} `}joinedComparisons(t){let n=Bt(t);return v(Vt,r=>{let i=e(t.slice(0,r.start()),``),a=t.slice(r.end());return n[r.start()]!==r.group(0)||!(_(Ht,i)||o(Ut,a)||_(Wt,i)&&o(Gt,a))?r.group(0):l.compared(r,r.group(0)===`<`?`less than`:`greater than`)},t)}signs(e){let t=l.namedLetters(e);if((t.includes(`<`)||t.includes(`>`))&&(t=this.joinedComparisons(t)),t=v(a(String.raw`(?<![\w-])(`+`XXXL|XXL|XXS|XL|XS|S|M|L)-(XXXL|XXL|XXS|XL|XS|S|M|L`+String.raw`)(?![\w-])`),e=>` ${q(e.group(1))} to ${q(e.group(2))} `,t),t=v(a(String.raw`#(?=\d)`),` number `,t),t=v(a(String.raw`#(?=[A-Z]{1,4}-?\d)`),` number `,t),t=v(a(String.raw`#(?=[A-Za-z])`),` hashtag `,t),t=v(a(String.raw`~(?=\s?\d)`),` about `,t),t=v(a(String.raw`(?<=[\w\s`+``+String.raw`])(>=|<=|!=)(?=[\w\s`+`])`),e=>l.compared(e,Ot[e.group(1)]),t),t=v(a(String.raw`(?<=\d) Ti\b`),` ${q(`TI`)}`,t),t=v(a(String.raw`\b([Rr])ef\.\s?[Nn]o\.(?=\s)`),`\\1eference number`,t),t.includes(`.`)){let e=t;t=v(a(String.raw`\b((?:[A-Za-z]\.){2,})`),t=>l.dotted(t,e),e)}{let e=t;t=v(a(String.raw`\b([A-Za-z]{1,6})\b(\.?)(?![A-Za-z'’])`),t=>this.abbreviation(t,e),e)}if(_(a(`\\d`),t)){t=v(Pn,e=>this.glued(e),t);let e=t;t=v(a(String.raw`(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])\.\d+`),t=>this.number(t,e),e)}t=v(a(`(?<=[A-Za-z])/(?=[A-Za-z])`),` `,t),t=v(a(String.raw`(?<=\d)`+``+String.raw`?\s?[<>]\s?(?=\d)|(?<=\s)[<>](?=\s)`),e=>l.compared(e,e.group(0).includes(`<`)?`less than`:`greater than`),t);let n=Object.keys(Dt);return t=v(RegExp(`[`+n.map(e=>e.replace(/[\\\]^[-]/g,`\\$&`)).join(``)+`]`,`gud`),e=>`≥≤≠`.includes(e.group(0))?l.compared(e,Dt[e.group(0)]):` ${Dt[e.group(0)]} `,t),t.replaceAll(``,``)}static unstretched(e){if(!_(a(String.raw`(.)\1\1`),e))return[];let t=(t,n)=>v(a(String.raw`(.)\1{2,}`),r=>r.group(1).repeat((r.end()===e.length?n:t)?2:1),e),n=!!_(a(String.raw`([lfsz])\1\1+$`),e)||new Set(e).size===1&&e[0]===`m`||v(a(String.raw`(.)\1{2,}$`),`\\1`,e)===`to`,r=[t(!0,n),t(!0,!n),t(!1,n),t(!1,!n)];return[...new Set(r)]}static chatWords(e,t){return e===`-`?[]:y(e).map(e=>e.startsWith(`*`)?N(e.slice(1).toLowerCase(),{raw:t,spell:!0}):N(e.toLowerCase(),{raw:e}))}static reply(e,t){let n=[];for(let r of[-1,1]){let i=t+r;for(;i>=0&&i<e.length&&e[i].kind===`punct`&&`"'()`.includes(e[i].mark);)i+=r;if(!(i>=0&&i<e.length))n.push(``);else if(e[i].kind===`word`)return!1;else{let t=e[i].mark;n.push(t);let a=i+r>=0&&i+r<e.length?e[i+r]:null;if(t===`,`&&a?.kind===`word`&&[...a.text].length===1&&a.text!==`i`)return!1}}return n.length!==2||n[0]!==`,`||n[1]!==`,`}wordTokens(e,t,i,o=!0,s=!1){let c=e.toLowerCase(),u=this.chat.get(c);if(u!==void 0&&([...e].length>1||r(e))&&!this.replies.has(c))return l.chatWords(u,e);if(Object.hasOwn(Yt,e)&&i&&f(i[0])){let t=Yt[e];return(qt.has(i.toLowerCase())?H(t):`the `+U(t)).split(` `).map(t=>N(t,{raw:e}))}let m=e.replaceAll(`'`,``),h=c.endsWith(`'s`),g=h?e.slice(0,-2):e;if([...g].length===1&&n(g)){let t=g===`A`&&!!i&&(!o||!s&&Jt.has(i.toLowerCase()));return[`a`,`A`,`I`,`i`].includes(g)&&!h&&!t?[N(g.toLowerCase(),{raw:e})]:[N(c,{raw:e,spell:!0})]}if(g.length>2&&`SsZz`.includes(g[0])&&new Set(c).size===1&&(t||!f(g)))return[N(c,{raw:e,phones:[[c[0],0]]})];if(f(g)&&n(g)&&g.length>=2&&g.length<=5&&(!t||!this.isWord(g.toLowerCase()))&&(kt.has(g)||!this.isWord(g.toLowerCase())))return[N(c,{raw:e,spell:!0})];if(!n(m))return[];let _=d(a(String.raw`[a-z]+(?:[A-Z][a-z]+)*[A-Z]{2,}(?:'s)?`),e);if(_||!this.isWord(c)){for(let t of _?[]:l.unstretched(c))if(this.isWord(t))return[N(t,{raw:e})];let t=p(a(String.raw`[A-Z]?[a-z']+|[A-Z]+(?![a-z])`),e);if(t.length>1&&t.join(``)===e){let e=[];for(let n of t){let t=_&&f(n)?kt.has(n):!1,r=n.length===1||t||f(n)&&!this.isWord(n.toLowerCase());if(_&&e.length&&e[e.length-1].spell&&(r||n===`'s`)){let t=e[e.length-1];e[e.length-1]={...t,text:t.text+n.toLowerCase(),raw:t.raw+n}}else e.push(N(n.toLowerCase(),{attach:e.length>0,raw:n,spell:r}))}return e}}return[N(c,{raw:e})]}tokens(e){let n=this.rewrite(e),s=p(a(`[A-Za-z]{2,}`),n),c=s.length>=3&&s.filter(e=>f(e)).length>=.6*s.length,u=[],d=a(`([A-Za-z ]+)|(.)`+String.raw`|('?[A-Za-z]+(?:'[A-Za-z]+)*)|(\d+)|(-)|(\S)`),h=!1,g=``,_=!0,v=[],b=[],ee=a(String.raw`\s*[A-Za-z]`),te=a(String.raw`[.!?…,;:—"')]*(?:\s|$)`),ne=a(String.raw`(?:[.?…,;—]*["']?(?:\s|$)|!+["']?\s*$)`);for(let e of t(d,n)){let[t,a,s,l,d,f]=e.groups(),p=e.start()?n[e.start()-1]:` `;if(d){h=u.length>0&&u[u.length-1].kind===`word`&&!i(p)&&e.end()<n.length&&m(n[e.end()]);continue}let x=[];if(t!==void 0)x=[N(t.replaceAll(` `,``).toLowerCase(),{raw:t,spell:!0})];else if(a!==void 0){let[e,t]=this.inline[a.codePointAt(0)-yt];x=[N(e.toLowerCase(),{raw:e,phones:t})]}else if(s!==void 0){let t=s;t.startsWith(`'`)&&!Xt.has(t.toLowerCase())?(u.push({kind:`punct`,mark:`'`}),t=t.slice(1)):t.startsWith(`'`)&&!this.isWord(t.toLowerCase())&&(t=t.slice(1)),x=this.wordTokens(t,c,g,o(ee,n.slice(e.end()))!==null,_),g=t,[`Ss`,`ss`,`Zz`,`zz`].includes(t)&&x.length===1&&(i(p)||`"'(`.includes(p))&&o(te,n.slice(e.end()))&&v.push(u.length),this.replies.has(t.toLowerCase())&&([...t].length>1||r(t))&&x.length===1&&(i(p)||`"'`.includes(p))&&o(ne,n.slice(e.end()))&&b.push(u.length)}else if(l!==void 0)x=y(K(l)).map(e=>N(e,{raw:l}));else if(f!==void 0&&ae.includes(f)){f===`'`&&m(p)&&e.end()<n.length&&m(n[e.end()])||u.push({kind:`punct`,mark:f}),`.!?;…`.includes(f)&&(_=!0),h=!1;continue}x.length&&h&&(x[0]={...x[0],attach:!0}),u.push(...x),h=!1,_&&=!x.length}for(let e of v){let t=e,n=e+1,r=e=>e.kind===`word`||`"'()`.includes(e.mark);for(;t&&r(u[t-1]);)--t;for(;n<u.length&&r(u[n]);)n+=1;if(u.slice(t,n).filter(e=>e.kind===`word`).every(e=>At.has(e.text)||e.phones!==null&&new Set(e.text).size===1)){let t=u[e];u[e]={...t,phones:[[t.text[0],0]]}}}for(let e of[...b].reverse())if(l.reply(u,e)){let t=u[e].raw;u.splice(e,1,...l.chatWords(this.chat.get(t.toLowerCase()),t))}return u}};const tr={ɑːɹ:`ɑː`,ɔːɹ:`ɔː`,oːɹ:`oː`,ɪɹ:`ɪ`,ɛɹ:`ɛ`,ʊɹ:`ʊ`},nr=/* @__PURE__ */ new Set([`ɚ`,`ɜː`,`aɪɚ`]),rr=/* @__PURE__ */ new Set([`and`,`or`,`if`]),ir=/* @__PURE__ */ new Set([`it`,`at`]);var X=class{word;phones;kind;final;attach;source;constructor(e,t,n=`c`,r=null,i=!1,a=``){this.word=e,this.phones=t,this.kind=n,this.final=r,this.attach=i,this.source=a}get lastWord(){let e=this.word.lastIndexOf(` `);return e<0?this.word:this.word.slice(e+1)}get firstWord(){let e=this.word.indexOf(` `);return e<0?this.word:this.word.slice(0,e)}};const Z=e=>e.map(([e])=>[e,0]);function Q(e){let t=e.map(([,e])=>e);if(t.includes(1))return[...e];let n;if(t.includes(2))n=t.lastIndexOf(2);else{let t=e.findIndex(([e])=>C.has(e));n=t<0?null:t}return e.map(([e,t],r)=>[e,r===n?1:t])}function ar(e){let t=!1;for(let n=e.length-1;n>=0;n--){let r=e[n],i=()=>r.final===null?Q(r.phones):r.final,a=n===e.length-1,o;o=r.kind===`u`?Z(r.phones):r.kind===`uf`?a?i():Z(r.phones):r.kind===`ur`?t?Z(r.phones):i():r.kind===`s`?t?r.phones:i():r.kind===`si`?t?n===0?r.phones:Z(r.phones):i():r.kind===`pf`&&a?i():r.phones,r.phones=[...o],t||=r.phones.some(([,e])=>e===1)}if(e.length&&!e.some(e=>e.phones.some(([,e])=>e))){for(let t of[...e].reverse())if(t.phones.some(([e])=>C.has(e))){t.phones=Q(t.phones);break}}}function or(e){for(let t=0;t+1<e.length;t++){let n=e[t],r=e[t+1];if(!n.phones.length||!r.phones.length||!C.has(r.phones[0][0]))continue;let[i,a]=n.phones[n.phones.length-1],o=n.phones.slice(0,-1);if(n.lastWord===`the`&&i===`ə`)n.phones=[...o,[`ɪ`,a]];else if(n.lastWord===`to`&&i===`ə`)n.phones=[...o,[`ʊ`,a]];else if(rr.has(r.firstWord))continue;else i in tr?n.phones=[...o,[tr[i],a],[`ɹ`,0]]:nr.has(i)?n.phones=[...n.phones,[`ɹ`,0]]:i===`t`&&ir.has(n.word)&&n.phones.length>1&&C.has(n.phones[n.phones.length-2][0])&&(n.phones=[...o,[`ɾ`,0]])}}function sr(e){let t=[];for(let n of e)n.attach&&t.length?t[t.length-1]=[...t[t.length-1],...n.phones]:t.push([...n.phones]);return t.filter(e=>e.length)}const cr=a(`(?<=[a-z]{2})leigh(?=s?$)`),lr=[`viruses`,`virus`];let $=null;var ur=class{lexicon=/* @__PURE__ */ new Map;aligner;model;homographs;functions=ze();phrases=Be();phraseLength;exceptions=He(`exceptions.txt`);names=Ue(`exceptions.txt`);nameStarts;nameLength;normalizer;cache=/* @__PURE__ */ new Map;constructor(e){for(let t of e.lexicon.split(/\r\n|\r|\n/)){if(!t)continue;let e=t.indexOf(`	`);this.lexicon.set(t.slice(0,e),t.slice(e+1))}this.aligner=new ee(e.align),this.model=e.g2p,this.homographs=e.homographs?new rt(e.homographs):null,this.phraseLength=Math.max(1,...[...this.phrases.keys()].map(e=>e.split(`\0`).length)),this.nameStarts=new Set([...this.names.keys()].map(e=>e.split(`\0`)[0])),this.nameLength=Math.max(1,...[...this.names.keys()].map(e=>e.split(`\0`).length)),this.normalizer=new er(e=>this.isWord(e))}entry(e,t){return t[0]===`phones`?t[1]:Me(e,t[1],this.aligner.align(e,t[1]))}known(e){let t=this.exceptions.get(e);if(t)return this.entry(e,t);let n=this.lexicon.get(e);return n===void 0?null:E(n)}ordinary(e){return e.length<3||![`s`,`es`,`ed`,`ing`].some(t=>this.lexicon.has(e+t))?null:this.known(e)}guess(e){if(this.model===null)return this.spell(e);let t=this.model.spellable(e),n=this.model.predict(t),r=e=>e>=0&&e<n.length&&`012`.includes(n[e].slice(-1));return n.forEach((e,t)=>{e===`R`&&!r(t-1)&&!r(t+1)&&(n[t]=`ER0`)}),n.some(e=>`012`.includes(e.slice(-1)))?Q(Me(t,n,this.aligner.align(t,n))):this.spell(e)}spell(e){$??=new Map([...We(`letters.txt`)].map(([e,t])=>[e,E(t)]));let t=e.toLowerCase().endsWith(`'s`),n=[...(t?e.slice(0,-2):e).toLowerCase()].filter(e=>$.has(e)).map(e=>$.get(e)),r=[];return n.forEach((e,t)=>{for(let[i,a]of e)r.push([i,a&&t<n.length-1?2:a])}),t&&r.length?M(r):r}isWord(e){return this.lexicon.has(e)||this.functions.has(e)||this.exceptions.has(e)}fromParts(e){if(e.endsWith(`'s`)&&e.length>3){let t=this.fromParts(e.slice(0,-2));return t&&[M(t[0]),t[1]]}if(_(cr,e))return this.word(v(cr,`ley`,e));for(let t of lr){let n=e.endsWith(t)&&e.length-t.length>2?this.known(t):null;if(n!==null){let[r,i]=this.word(e.slice(0,-t.length));return[Q(Pe(_t(r,n))),i===`model`?`model`:`ending`]}}return null}word(e){let t=this.cache.get(e);if(t)return t;let n=this.known(e),r=`dictionary`;if(n===null&&(n=vt(e,e=>this.known(e),e=>this.ordinary(e)),r=`ending`,n!==null&&(n=Pe(n))),n===null){let t=this.fromParts(e);t&&([n,r]=t)}n===null&&e.endsWith(`'s`)&&e.length>3&&(n=M(this.guess(e.slice(0,-2))),r=`model`),n===null&&(n=this.guess(e),r=`model`);let i=[n,r];return this.cache.set(e,i),i}units(e,t,n){let r=[],i=0;for(;i<e.length;){let a=e[i];if(a.phones!==null)r.push(new X(a.text,a.phones,`p`,null,a.attach,`inline`));else if(a.spell)r.push(new X(a.text,this.spell(a.text),`p`,null,a.attach,`letters`));else{let o=this.nameStarts.has(a.text)?this.name(e,i):0;if(o){let t=e.slice(i,i+o).map(e=>e.text),n=this.names.get(l(t));t.forEach((e,t)=>r.push(new X(e,this.entry(e,n[t]),`p`,null,a.attach&&t===0,`name`))),i+=o;continue}if(o=this.phrase(e,i),o){let t=e.slice(i,i+o).map(e=>e.text),[n,s,c]=this.phrases.get(l(t));s.length===1?r.push(new X(t.join(` `),s[0],n,c,a.attach,`phrase`)):t.forEach((e,t)=>r.push(new X(e,s[t],n,null,a.attach&&t===0,`phrase`))),i+=o;continue}if(this.homographs&&n.length&&this.homographs.covers(a.text)&&!this.exceptions.has(a.text)){let[e,o]=this.homographs.pronounce(a.text,n,t[i],a.attach),s=o?M(E(e)):E(e);r.push(new X(a.text,s,`c`,null,a.attach,`homograph`))}else if(this.functions.has(a.text)){let[e,t,n]=this.functions.get(a.text);r.push(new X(a.text,t,e,n,a.attach,`function`))}else{let[e,t]=this.word(a.text);r.push(new X(a.text,e,`c`,null,a.attach,t))}}i+=1}return r}name(e,t){for(let n=Math.min(this.nameLength,e.length-t);n>1;n--){let r=e.slice(t,t+n);if(this.names.has(l(r.map(e=>e.text)))&&r.every(e=>e.phones===null&&!e.spell)&&!r.slice(1).some(e=>e.attach))return n}return 0}phrase(e,t){for(let n=Math.min(this.phraseLength,e.length-t);n>1;n--){let r=e.slice(t,t+n);if(r.slice(1).every(e=>e.phones===null&&!e.spell&&!e.attach)&&r[0].phones===null&&!r[0].spell&&this.phrases.has(l(r.map(e=>e.text))))return n}return 0}analyse(e){let t=this.normalizer.tokens(e),n=t.map(e=>e.kind===`punct`?e.mark:e.raw||e.text||`x`),r=[],i=[],a=[],o=()=>{if(i.length){let e=this.units(i,a,n);ar(e),or(e),r.push([`clause`,e]),i=[],a=[]}};return t.forEach((e,t)=>{e.kind===`punct`?(o(),r.push([`punct`,e.mark])):(i.push(e),a.push(t))}),o(),r}pieces(e){return this.analyse(e).map(e=>e[0]===`punct`?[!0,e[1]]:[!1,sr(e[1])])}phonemize(e){if(!y(e).length)throw new de(`text must not be empty`);return fe(this.pieces(e))}tryPhonemize(e){try{return this.phonemize(e)}catch(e){if(e instanceof de)return null;throw e}}};export{ur as Phonemizer};