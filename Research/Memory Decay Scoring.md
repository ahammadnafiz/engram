# **Memory Decay Scoring: Complete Formula Reference**
## **Production & Research-Driven Formulas (2024-2025)**

---

## **TABLE OF CONTENTS**

1. **Classical Memory Models**
2. **Production System Formulas**
3. **Advanced Research Models**
4. **Social Platform Ranking Algorithms**
5. **Hybrid & Composite Models**
6. **Implementation Guidelines**

---

## **1. CLASSICAL MEMORY MODELS**

### **1.1 Ebbinghaus Forgetting Curve (1885)**

The foundational exponential decay model:

**Formula:**
```
R = e^(-t/S)
```

**Where:**
- `R` = Memory retention (0 to 1)
- `t` = Time elapsed since learning
- `S` = Memory strength
- `e` = Euler's number (≈2.71828)

**Characteristics:**
- Memory retention decreases over time with a steeper decline initially and slower rate later
- Forms the basis for most modern memory decay implementations

---

### **1.2 Power Law Forgetting**

**Formula:**
```
R = t^(-α)
```

**Where:**
- `α` = Decay exponent (typically 0.3-0.5)
- `t` = Time since encoding

**Characteristics:**
- Forgetting is exponential in nature with decay rate proportional to sample size and constant probability of single particle decay
- Better fits long-term memory patterns

---

### **1.3 Two-Phase Decay Model**

Combines exponential and power-law phases representing fast (linear) and slow (nonlinear) decay dynamics respectively

**Formula:**
```
M(t) = {
  C₁ · e^(-α·t)           if t ≤ t_switch
  C₂ · t^(-β)             if t > t_switch
}
```

**Where:**
- `C₁, C₂` = Scaling constants
- `α` = Exponential decay rate
- `β` = Power-law decay exponent
- `t_switch` = Switching point (typically 10-11 days)

**Application:**
- Wikipedia page views
- Collective memory studies
- Decay phase shifts typically occur 10-11 days after peak across event categories

---

### **1.4 Biexponential Decay**

Universal pattern for cultural product attention decay combining communicative memory (fast) and cultural memory (slow)

**Formula:**
```
A(t) = C₁·e^(-α·t) + C₂·e^(-β·t)
```

**Where:**
- `α > β` (fast and slow decay rates)
- Different retention times by content type:
  - Music: ~5.6 years
  - Biographies: 20-30 years

---

## **2. PRODUCTION SYSTEM FORMULAS**

### **2.1 MemoryBank (2024) - Standard Implementation**

**Core Formula:**
```
R = e^(-t/S)

S_initial = 1
S_new = S_old + 1  (on recall)
t_new = 0          (on recall)
```

**Weighted Memory Retrieval:**
```
score = (0.6 × relevance) + (0.25 × recency) + (0.15 × importance)
```

Memory strength S is discrete, initialized at 1 upon first mention, increased by 1 when recalled, and time t reset to 0, reducing forgetting probability for frequently accessed memories

---

### **2.2 MemoryBank Hourly Decay**

**Recency Score Formula:**
```
recency_score = 0.995^hours_elapsed
```

**Characteristics:**
- Decay factor: 0.995 per hour
- After 24 hours: ~0.886 retention
- After 168 hours (1 week): ~0.433 retention
- After 720 hours (30 days): ~0.025 retention

**Implementation:**
```python
def calculate_memory_score(memory, current_time, query_embedding):
    # Time decay
    hours = (current_time - memory.timestamp).total_seconds() / 3600
    recency_score = 0.995 ** hours
    
    # Semantic relevance
    relevance_score = cosine_similarity(
        memory.embedding, 
        query_embedding
    )
    
    # Importance (LLM-generated 0-1 scale)
    importance_score = memory.importance_factor
    
    # Weighted combination
    final_score = (
        0.6 * relevance_score + 
        0.25 * recency_score + 
        0.15 * importance_score
    )
    
    return final_score
```

---

### **2.3 Alternative Recency Formulas**

**Hyperbolic Decay:**
```
recency_score = 1.0 / (1.0 + decay_rate × hours)
```
- Common: `decay_rate = 0.01`
- Slower initial decay than exponential

**Linear 30-Day Decay:**
```
recency_score = max(0, 1 - (days_elapsed / 30))
```
- Simple implementation
- Zero score after 30 days

**Logarithmic Decay:**
```
recency_score = 1.0 / log(1 + hours)
```
- Very gradual decay
- Never reaches zero

---

### **2.4 Exponential Moving Average (EMA)**

Exponential decay applies weight assignments where if decay factor α=0.5 and window is one day, an event now has weight 1.0, one day old has 0.5, two days old has 0.25

**Formula:**
```
weight(t) = α^t
```

**Where:**
- `α` = Decay factor (0 < α < 1)
- `t` = Time periods elapsed

**Smoothed Score:**
```
EMA_t = α × value_t + (1 - α) × EMA_(t-1)
```

---

## **3. ADVANCED RESEARCH MODELS**

### **3.1 Time-Based Resource-Sharing (TBRS) Model**

During distractor processing all items suffer temporal decay described by exponential function, plus interference-based degradation of last-presented item

**Decay Function:**
```
strength(t) = strength_0 × e^(-λ·t)
```

**With Interference:**
```
strength(t) = [strength_0 × e^(-λ·t)] × (1 - interference_factor)
```

**Where:**
- `λ` = Decay rate constant
- `interference_factor` = 0 to 1

---

### **3.2 Stretched Exponential (Kohlrausch Function)**

**Formula:**
```
R(t) = e^(-(t/τ)^β)
```

**Where:**
- `τ` = Characteristic time
- `β` = Stretching exponent (0 < β ≤ 1)
  - β = 1: Simple exponential
  - β < 1: Slower than exponential (sub-exponential)

**Application:**
- Glass relaxation dynamics
- Daily page views of academic articles

---

### **3.3 Wilson Score Interval (Confidence Sort)**

Used for rating systems with upvotes/downvotes:

**Formula:**
```
score = (p̂ + z²/2n - z√[(p̂(1-p̂) + z²/4n)/n]) / (1 + z²/n)
```

**Where:**
- `p̂` = proportion of upvotes = upvotes/(upvotes + downvotes)
- `n` = total votes
- `z` = z-score (1.96 for 95% confidence)

**Simplified Implementation:**
```python
def wilson_score(upvotes, downvotes):
    n = upvotes + downvotes
    if upvotes == 0:
        return 0
    
    phat = upvotes / n
    z = 1.96  # 95% confidence
    
    numerator = (phat + z*z/(2*n) - 
                 z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n))
    denominator = 1 + z*z/n
    
    return numerator / denominator
```

---

## **4. SOCIAL PLATFORM RANKING ALGORITHMS**

### **4.1 Hacker News Algorithm**

**Basic Formula:**
```
score = (votes - 1) / (hours_age + 2)^gravity
```

**Standard Parameters:**
- `gravity = 1.8`
- `hours_age` = time since submission
- Penalty factor applied for various flags

Score decreases as time increases with faster decrease for older items if gravity increased, and 24-hour old items have very low scores regardless of votes

**Enhanced Version with Penalties:**
```
score = (votes^0.8) / ((hours_age + 2)^1.8) × penalty_factor
```

**Penalty Factors:**
- Normal: 1.0
- Controversial (>40 comments): 0.2-0.4
- Flagged content: 0.8-0.9
- Voting ring detected: 0.001

**Implementation:**
```python
def hacker_news_score(votes, hours_age, gravity=1.8):
    return (votes - 1) / pow(hours_age + 2, gravity)
```

---

### **4.2 Reddit Hot Algorithm**

Reddit's hot ranking uses logarithm function to weight first votes higher than rest, where first 10 upvotes have same weight as next 100 which have same weight as next 1000

**Formula:**
```
score = sign(s) × log₁₀(max(|s|, 1)) + (epoch_seconds / 45000)
```

**Where:**
- `s = upvotes - downvotes`
- `sign(s)` = {-1, 0, 1}
- `epoch_seconds` = seconds since epoch (adjusted by baseline)
- `45000` = 12.5 hour interval

**Key Features:**
- Logarithmic vote weighting
- Time-based boost (not penalty)
- 10-fold increase in points equates to being submitted 12.5 hours later, so 1-hour-old post must improve vote differential 10x over next 12.5 hours to maintain rating

**Implementation:**
```python
def reddit_hot_score(upvotes, downvotes, submission_date):
    s = upvotes - downvotes
    order = math.log10(max(abs(s), 1))
    
    if s > 0:
        sign = 1
    elif s < 0:
        sign = -1
    else:
        sign = 0
    
    epoch = datetime(1970, 1, 1).timestamp()
    seconds = submission_date.timestamp() - epoch
    seconds -= 1134028003  # Reddit epoch adjustment
    
    return round(sign * order + seconds / 45000.0, 7)
```

---

### **4.3 Reddit Confidence Sort (Wilson Score)**

Used for comment ranking:

Treats vote count as statistical sampling of hypothetical full vote by everyone, placing comments with 10 upvotes and 1 downvote above those with 40 upvotes and 20 downvotes based on confidence

**Advantage:**
- Submission time irrelevant
- Prevents early comment bias
- Self-correcting (wrong predictions get more data from top placement)

---

## **5. HYBRID & COMPOSITE MODELS**

### **5.1 Multi-Factor Weighted Score**

**General Formula:**
```
score = Σ(weight_i × factor_i)
```

**Common Configurations:**

**Configuration A: Balanced**
```
score = 0.4×relevance + 0.3×recency + 0.2×importance + 0.1×frequency
```

**Configuration B: Relevance-Focused**
```
score = 0.6×relevance + 0.25×recency + 0.15×importance
```

**Configuration C: Time-Sensitive**
```
score = 0.35×relevance + 0.45×recency + 0.20×importance
```

---

### **5.2 Exponential Decay with Boost**

**Formula:**
```
score = base_score × e^(-λ·t) + boost_factor
```

**Boost Conditions:**
- Recent access: `boost = 0.2`
- High importance: `boost = 0.15`
- User preference match: `boost = 0.1`

---

### **5.3 Adaptive Decay Rate**

**Formula:**
```
decay_rate = base_rate × (1 + context_factor)

context_factor = {
  0.5   if high_importance
  1.0   if normal
  2.0   if low_importance
}

score = initial_score × e^(-decay_rate × t)
```

---

## **6. IMPLEMENTATION GUIDELINES**

### **6.1 Choosing the Right Formula**

| Use Case | Recommended Formula | Decay Half-Life |
|----------|-------------------|-----------------|
| Short-term chat memory | Exponential (λ=0.005/hr) | ~139 hours |
| Long-term user profile | Biexponential | Varies by component |
| Hot content ranking | HN/Reddit algorithm | 12-24 hours |
| Confidence-based ranking | Wilson Score | N/A |
| Memory consolidation | MemoryBank | Configurable |

---

### **6.2 Parameter Tuning Guidelines**

**Exponential Decay Rate Selection:**
```
λ = ln(2) / desired_half_life

Examples:
- 1 day half-life:   λ = 0.693 / 24 = 0.029/hr
- 1 week half-life:  λ = 0.693 / 168 = 0.004/hr
- 1 month half-life: λ = 0.693 / 720 = 0.001/hr
```

**Recency Weight Optimization:**
1. Start with 0.995^hours (MemoryBank standard)
2. Adjust based on domain:
   - Fast-changing: 0.99^hours
   - Stable content: 0.998^hours

---

### **6.3 Production Implementation Patterns**

**Pattern 1: Precomputed Scores**
```python
class MemoryItem:
    def __init__(self):
        self.base_score = 0.0
        self.last_computed = None
        self.decay_rate = 0.995
    
    def get_score(self, current_time):
        if self.last_computed is None:
            return self.base_score
        
        hours = (current_time - self.last_computed).total_seconds() / 3600
        decay_factor = self.decay_rate ** hours
        return self.base_score * decay_factor
    
    def update_score(self, new_base_score, current_time):
        self.base_score = new_base_score
        self.last_computed = current_time
```

**Pattern 2: Batch Decay Application**
```python
def apply_batch_decay(memory_items, current_time, decay_rate=0.995):
    for item in memory_items:
        hours = (current_time - item.timestamp).total_seconds() / 3600
        item.score *= decay_rate ** hours
    
    return sorted(memory_items, key=lambda x: x.score, reverse=True)
```

**Pattern 3: Tiered Memory System**
```python
class TieredMemory:
    def __init__(self):
        self.hot_memory = []    # < 1 day, no decay
        self.warm_memory = []   # 1-7 days, slow decay
        self.cold_memory = []   # > 7 days, fast decay
    
    def calculate_score(self, item, current_time):
        age_days = (current_time - item.timestamp).days
        
        if age_days < 1:
            return item.base_score  # No decay
        elif age_days < 7:
            return item.base_score * 0.998 ** (age_days * 24)
        else:
            return item.base_score * 0.995 ** (age_days * 24)
```

---

### **6.4 Common Pitfalls & Solutions**

**Pitfall 1: Score Inflation Over Time**
- **Problem:** Newer items always dominate
- **Solution:** Normalize scores within time windows

**Pitfall 2: Cliff Effect**
- **Problem:** Sharp cutoffs create discontinuities
- **Solution:** Use smooth transitions (sigmoid, tanh)

**Pitfall 3: Computational Overhead**
- **Problem:** Recalculating all scores frequently
- **Solution:** Lazy evaluation + caching + batch updates

**Pitfall 4: Cold Start**
- **Problem:** New items have no history
- **Solution:** Initial boost period or default high score

---

### **6.5 Testing & Validation**

**Unit Test Template:**
```python
def test_memory_decay():
    # Test 1: Basic decay
    assert calculate_decay(1.0, 0, 0.995) == 1.0
    assert calculate_decay(1.0, 24, 0.995) < 0.9
    
    # Test 2: Recall resets decay
    memory = MemoryItem()
    memory.recall()
    assert memory.time_since_access == 0
    
    # Test 3: Importance affects decay rate
    important = calculate_score(high_importance=True)
    normal = calculate_score(high_importance=False)
    assert important > normal
```

---

## **7. QUICK REFERENCE TABLE**

| Formula Type | Expression | Best For | Decay Rate |
|-------------|-----------|----------|------------|
| Simple Exponential | `e^(-λt)` | General purpose | Configurable |
| MemoryBank | `0.995^hours` | Chat memory | 0.5% per hour |
| Power Law | `t^(-α)` | Long-term trends | Slow |
| Biexponential | `C₁e^(-αt) + C₂e^(-βt)` | Cultural memory | Dual-phase |
| Hacker News | `(v-1)/(t+2)^1.8` | Content ranking | Fast |
| Reddit Hot | `log₁₀(v) + t/45000` | Social ranking | Logarithmic |
| Wilson Score | `(p̂ + z²/2n - ...)/(1 + z²/n)` | Ratings | None (statistical) |

---

## **SUMMARY**

This comprehensive reference covers:
- **7 classical models** from cognitive science
- **4 production implementations** with code
- **3 social platform algorithms** with real-world usage
- **5 hybrid approaches** for complex systems
- **Complete implementation guidelines** with patterns

The choice of formula depends on:
1. **Time scale** (hours, days, months)
2. **Content type** (transient vs. persistent)
3. **User behavior** (frequency of access)
4. **System constraints** (computational resources)

**Recommended starting point:** MemoryBank's `0.995^hours` with weighted scoring provides excellent balance between accuracy and simplicity for most AI memory applications.