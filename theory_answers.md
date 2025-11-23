
# Theory Answers — Assignment 4: Advanced Image Generation

This file contains the written responses for **Part 2 (Diffusion Models)** and **Part 3 (Energy-Based Models)** of the assignment.

---

# 2. Theory: Building Blocks of a Diffusion Model

---

## **1. Sinusoidal Time Embedding Formula**

For timestep \( t \) and embedding dimension \( d \):

\[
\text{emb}_i(t) =
\begin{cases}
\sin\left(t / 10000^{\frac{i}{d}}\right), & i \text{ even} \\
\cos\left(t / 10000^{\frac{i-1}{d}}\right), & i \text{ odd}
\end{cases}
\]

---

## **2. Embedding for t = 1, dimension = 8**

Computed values:

\[
[0.84147,\; 0.54030,\; 0.09983,\; 0.99500,\;
0.0099998,\; 0.99995,\; 0.00099999,\; 0.9999995]
\]

---

## **3. Relation to Transformer Positional Encodings**

**Similarity:**  
Both diffusion and transformer models use sinusoidal embeddings with multi-frequency components.

**Difference:**  
- Transformers encode **token position**.  
- Diffusion models encode **timestep / noise level**, conditioning every UNet block.  

---

## **4. Downsampling Resolution**

Three stride-2 convolutions:

\[
64 \rightarrow 32 \rightarrow 16 \rightarrow 8
\]

Final resolution: **8 × 8**.

---

## **5. UNet Output and Loss**

The UNet predicts the added noise:

\[
\hat{\epsilon}_\theta(x_t, t)
\]

Training loss:

\[
\mathcal{L} = \mathbb{E}\left[\| \epsilon - \hat{\epsilon}_\theta(x_t, t) \|^2\right]
\]

---

# 3. Theory: Building Blocks of an Energy Model

---

## **6. Basic Gradient Calculations**

Given:

\[
y = x^2 + 3x
\]

### a) Gradient at \( x = 2 \):

\[
\frac{dy}{dx} = 2x + 3 = 7
\]

### b) If `requires_grad=False`:

- No gradient tracking  
- `x.grad` remains `None`

### c) If omitted:

Default is `False`, so gradients are not tracked.

---

## **7. Gradients with Learnable Weights**

Given:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
```

### a) Why is `w.grad = None`?

Because `w` was created without `requires_grad=True`.

### b) Version that computes gradient:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)

y = w[0] * x**2 + w[1] * x
y.backward()

# w.grad = [4, 2]
```

### c) If omitted:

No gradient will be tracked.

---

## **8. Breaking the Graph with detach()**

Detaching:

```python
z = y.detach()
```

removes `z` from the graph → gradients cannot flow back.

### Fix while keeping z:

```python
z = y.detach().requires_grad_(True)
```

---

## **9. Gradient Accumulation**

Example:

```python
x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
y2 = x * 3
y2.backward()
```

Result:

```
x.grad = 2 + 3 = 5
```

### Avoid accumulation:

```python
x.grad = None
# or
x.grad.zero_()
```

---

# End of Theory Answers
