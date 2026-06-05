#!/usr/bin/env python3
"""
SnapBench: text perturbation generator (8 types, entity-targeted).

Pre-computed perturbations are already stored in snap_bench.json.
Use this script only if you need to regenerate them.

Usage:
  python benchmark/gen_text_perturbations.py --gpu 0 --chunk-in chunk_0.json --chunk-out result_0.json
"""
import json, os, re, random, argparse

# ── QWERTY 邻键映射 ──
QWERTY_MAP = {
    'a': 'sqwzx', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wrsdf',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'uojkl', 'j': 'huiknm',
    'k': 'jiolm', 'l': 'kopi', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'awedxz', 't': 'ryfg',
    'u': 'yhij', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tugh',
    'z': 'asx',
}

# ── sent_add 模板库（去掉过短模板，保留 20 字以上）──
SENT_ADD_TEMPLATES = [
    "Traffic was bad today.", "Need to buy groceries later.",
    "Slept terribly last night.", "Skipped breakfast today.", "Running late as usual.",
    "Laundry day tomorrow.",
    "Just got back from the gym, super tired.", "Had pasta for lunch and now feeling sleepy.",
    "Stuck in traffic for an hour, so annoying.", "Woke up late and missed the bus again.",
    "My neighbor's dog was barking all night.", "Forgot my umbrella and got soaked in rain.",
    "Spent the whole morning cleaning the kitchen.", "My roommate ate all the leftovers again.",
    "Been waiting in line for twenty minutes now.", "Just finished a two-hour meeting, exhausted.",
    "The WiFi keeps dropping every five minutes.", "Tried a new restaurant downtown, food was okay.",
    "My flight got delayed by three hours.", "Lost my keys again, third time today.",
    "The elevator is broken so took the stairs.", "Weather is nice today.",
    "Should probably go for a walk later.", "My phone battery is almost dead.",
    "Really need to clean my room this weekend.", "The coffee machine at work is broken.",
    "Watched a movie last night, it was boring.", "Can't find my charger anywhere.",
    "The train was packed this morning.", "My alarm didn't go off again.",
    "Thinking about what to have for dinner.", "Just paid my electricity bill.",
    "The air conditioning is way too cold.", "Ran out of toothpaste this morning.",
    "My headphones stopped working yesterday.", "The supermarket was closed when I got there.",
    "Need to renew my driver's license soon.", "Forgot to water the plants again.",
    "The meeting got rescheduled to next week.", "My shoes are still wet from the rain.",
    "The new season of that show just came out.", "Should really start exercising more.",
    "The parking lot was completely full.", "My laptop is running so slow today.",
    "The food delivery took over an hour.", "Just got a haircut, looks terrible.",
    "Need to fix the leaking faucet.", "The dishwasher is making weird noises.",
    "Waiting for the repairman to show up.", "My subscription expired yesterday.",
    "The store didn't have what I needed.", "Tried cooking a new recipe, total disaster.",
    "The printer ran out of ink again.", "My glasses prescription needs updating.",
    "The bus was ten minutes late today.", "Dropped my phone and cracked the screen.",
    "The vending machine ate my coin.", "Need to schedule a dentist appointment.",
    "The hallway light has been flickering.", "My umbrella broke in the wind.",
    "The checkout line was ridiculously long.", "Just realized I forgot my wallet.",
    "The hot water heater is acting up.", "My package still hasn't arrived.",
    "The office was freezing cold today.", "Spilled coffee on my shirt this morning.",
    "The garbage truck woke me up at six.", "My internet has been so slow lately.",
    "The microwave made a spark yesterday.", "Need to return those shoes I bought.",
    "The smoke detector kept beeping at night.", "My bicycle tire is flat again.",
    "The ATM was out of service.", "Forgot my lunch at home today.",
    "The crosswalk signal was broken.", "My watch battery died last week.",
    "The subway was delayed for thirty minutes.", "Just tripped on the sidewalk.",
    "The gas station nearby just closed.", "My jacket zipper is stuck.",
    "The library book is overdue.", "Left the oven on by accident.",
    "The doorbell hasn't worked in months.", "My shoelace broke this morning.",
    "The ceiling fan is making clicking sounds.", "Need to buy new light bulbs.",
    "The car wash was closed today.", "My pen ran out of ink during class.",
    "The power went out for ten minutes.", "Just stepped in a puddle.",
    "The fire alarm went off for no reason.", "My socks don't match today.",
    "The water pressure is so low.", "Need to take the car in for service.",
    "The ice maker stopped working.", "My backpack strap just broke.",
    "The road construction is never ending.", "Overslept by two hours.",
    "The fridge is making a humming noise.", "Just burned my tongue on hot soup.",
    "The window won't close properly.", "My contact lens keeps falling out.",
    "The dryer ate one of my socks.", "Need to call the plumber soon.",
    "The streetlight outside is broken.", "My calculator needs new batteries.",
    "The escalator was out of order.", "Forgot to set my alarm last night.",
    "The paint on the wall is peeling.", "My stapler jammed again.",
    "The toilet keeps running.", "Just missed the last bus home.",
]

# ── 扰动函数 ──

def _stem_match(word, entity_word):
    """检查 word 是否是 entity_word 的变形（复数/词干等）"""
    w = word.lower().strip('?.,!;:')
    e = entity_word.lower()
    if w == e: return True
    # 复数变形
    if w == e + 's' or w == e + 'es': return True
    if w.endswith('ies') and e.endswith('y') and w[:-3] == e[:-1]: return True  # berries/berry
    if w.endswith('ves') and e.endswith('f') and w[:-3] == e[:-1]: return True  # knives/knife
    if e == w + 's' or e == w + 'es': return True
    # 词干
    if w.startswith(e) and len(w) - len(e) <= 3: return True
    if e.startswith(w) and len(e) - len(w) <= 3: return True
    return False

def find_entity_in_query(words, entity):
    """找到 entity 在 query 中的位置（返回起止 index）"""
    entity_words = entity.lower().split()
    # 精确匹配
    for i in range(len(words)):
        match = True
        for j, ew in enumerate(entity_words):
            if i + j >= len(words):
                match = False; break
            if words[i+j].lower().strip('?.,!') != ew:
                match = False; break
        if match:
            return i, i + len(entity_words)
    # 词干匹配（处理复数等变形）
    for i in range(len(words)):
        match = True
        for j, ew in enumerate(entity_words):
            if i + j >= len(words):
                match = False; break
            if not _stem_match(words[i+j], ew):
                match = False; break
        if match:
            return i, i + len(entity_words)
    # fallback: 找包含 entity 最后一个词（含词干）的位置
    last_ew = entity_words[-1] if entity_words else ''
    for i, w in enumerate(words):
        if _stem_match(w, last_ew):
            return i, i + 1
    return None, None

def get_entity_word(words, entity):
    """获取 query 中 entity 对应的实际词（保留原始大小写和标点）"""
    start, end = find_entity_in_query(words, entity)
    if start is None:
        return None, None
    return start, ' '.join(words[start:end])

def char_add(text, entity, rng):
    """在 entity 词中随机位置插入 1 个随机字母"""
    words = text.split()
    idx, _ = find_entity_in_query(words, entity)
    if idx is None: return None
    word = words[idx]
    core = word.strip('?.,!;:')
    if len(core) < 2: return None
    pos = rng.randint(1, len(core) - 1)
    letter = rng.choice('abcdefghijklmnopqrstuvwxyz')
    new_core = core[:pos] + letter + core[pos:]
    words[idx] = word.replace(core, new_core)
    return ' '.join(words)

def char_delete(text, entity, rng):
    """从 entity 词中删除 1 个字母"""
    words = text.split()
    idx, _ = find_entity_in_query(words, entity)
    if idx is None: return None
    word = words[idx]
    core = word.strip('?.,!;:')
    if len(core) < 2: return None  # 至少 2 字母才能删
    pos = rng.randint(0, len(core) - 1)
    new_core = core[:pos] + core[pos+1:]
    words[idx] = word.replace(core, new_core)
    return ' '.join(words)

def char_change(text, entity, rng):
    """将 entity 词中 1 个字母替换为 QWERTY 邻键"""
    words = text.split()
    idx, _ = find_entity_in_query(words, entity)
    if idx is None: return None
    word = words[idx]
    core = word.strip('?.,!;:')
    candidates = [(i, c) for i, c in enumerate(core.lower()) if c in QWERTY_MAP]
    if not candidates: return None
    pos, ch = rng.choice(candidates)
    new_ch = rng.choice(QWERTY_MAP[ch])
    new_core = core[:pos] + (new_ch.upper() if core[pos].isupper() else new_ch) + core[pos+1:]
    words[idx] = word.replace(core, new_core)
    result = ' '.join(words)
    return result if result != text else None

def char_swap(text, entity, rng):
    """交换 entity 词中 1 对相邻字母"""
    words = text.split()
    idx, _ = find_entity_in_query(words, entity)
    if idx is None: return None
    word = words[idx]
    core = word.strip('?.,!;:')
    if len(core) < 2: return None
    positions = [i for i in range(len(core)-1) if core[i] != core[i+1]]
    if not positions: return None
    pos = rng.choice(positions)
    new_core = core[:pos] + core[pos+1] + core[pos] + core[pos+2:]
    if new_core == core: return None
    words[idx] = word.replace(core, new_core)
    return ' '.join(words)

def _ensure_question_mark(result, original):
    """确保结果以原句的末尾标点结尾（通常是?），去掉中间的多余标点"""
    # 找原句末尾标点
    trail = ''
    for ch in reversed(original):
        if ch in '?!.':
            trail = ch
            break
    if not trail:
        return result
    # 去掉所有 ? 然后加回末尾
    result = result.rstrip('?!. ')
    # 去掉句中因为 swap/repeat 产生的中间标点
    import re as _re
    result = _re.sub(r'([?!.])(?=\s)', '', result).strip()
    return result + trail

def word_repeat(text, entity, rng):
    """重复 entity 词 1 次"""
    words = text.split()
    start, end = find_entity_in_query(words, entity)
    if start is None: return None
    entity_span = words[start:end]
    clean = [w.strip('?.,!;:') for w in entity_span]
    new_words = words[:end] + clean + words[end:]
    result = ' '.join(new_words)
    return _ensure_question_mark(result, text)

def word_swap(text, entity, rng):
    """将 entity 词与随机左或右邻词交换"""
    words = list(text.split())
    start, end = find_entity_in_query(words, entity)
    if start is None: return None
    # 随机选方向
    can_left = start > 0
    can_right = end < len(words)
    if not can_left and not can_right: return None
    if can_left and can_right:
        direction = rng.choice(['left', 'right'])
    elif can_left:
        direction = 'left'
    else:
        direction = 'right'
    
    if direction == 'left':
        # entity span 和左边一个词交换
        left_word = words[start-1]
        entity_span = words[start:end]
        words = words[:start-1] + entity_span + [left_word] + words[end:]
    else:
        # entity span 和右边一个词交换
        right_word = words[end]
        entity_span = words[start:end]
        words = words[:start] + [right_word] + entity_span + words[end+1:]
    
    result = ' '.join(words)
    result = _ensure_question_mark(result, text)
    return result if result != text else None

def sent_add(text, entity, rng):
    """随机在问句前或后加 1 句无关闲聊"""
    template = rng.choice(SENT_ADD_TEMPLATES)
    position = rng.choice(['before', 'after'])
    if position == 'before':
        return f"{template} {text}"
    else:
        # 后缀：去掉原句问号，加闲聊句
        base = text.rstrip('?!.')
        return f"{base}? {template}"

def sent_replace(text, entity, rng):
    """将问句完全泛化为最通用形式，删掉所有类别词，只保留疑问词结构。"""
    text_lower = text.strip().lower()

    # 根据疑问词返回最简通用句
    if text_lower.startswith('where'):
        return "Where is this?"
    if text_lower.startswith('who'):
        return "Who is this?"
    if text_lower.startswith('which'):
        return "Which one is this?"
    if text_lower.startswith('how'):
        return "How is this?"
    if text_lower.startswith('is '):
        return "Is this it?"

    # 所有 what 开头的，统一变成 "What is this?"
    return "What is this?"

# ── 主逻辑 ──

PERTURB_FNS = {
    'char_add': char_add,
    'char_delete': char_delete,
    'char_change': char_change,
    'char_swap': char_swap,
    'word_repeat': word_repeat,
    'word_swap': word_swap,
    'sent_add': sent_add,
    'sent_replace': sent_replace,
}

def qa_check(original, perturbed):
    """验证扰动结果有效"""
    if not perturbed: return False
    if perturbed.strip() == original.strip(): return False
    if len(perturbed.strip()) < 3: return False
    return True

def generate_perturbations(en_query, entity, seed):
    """生成 8 种扰动"""
    results = {}
    for pt_name, fn in PERTURB_FNS.items():
        rng = random.Random(seed + hash(pt_name))
        result = fn(en_query, entity, rng)
        if qa_check(en_query, result):
            results[pt_name] = {"perturbed_text": result, "changed": True}
        else:
            results[pt_name] = {"perturbed_text": None, "changed": False}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--chunk-in", required=True)
    parser.add_argument("--chunk-out", required=True)
    args = parser.parse_args()
    
    from tqdm import tqdm
    chunk = json.load(open(args.chunk_in, encoding='utf-8'))
    print(f"[GPU {args.gpu}] {len(chunk)} 条", flush=True)
    
    for item in tqdm(chunk, desc=f"GPU{args.gpu}"):
        text = item.get('query_text', item.get('query_text_en', '')).strip()
        entity = (item.get('query_entity') or '').strip()
        seed = hash(item['query_id']) & 0xFFFFFFFF
        
        perturbs = generate_perturbations(text, entity, seed)
        item['text_perturbations'] = perturbs
    
    with open(args.chunk_out, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    print(f"[GPU {args.gpu}] 完成 → {args.chunk_out}", flush=True)

if __name__ == '__main__':
    main()
