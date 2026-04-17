# PreBuyPc — Django App Developer Specification
**Version:** 1.0  
**Project Type:** ML-powered laptop recommendation web app  
**Stack:** Python 3.11 · Django 5.x · PostgreSQL · XGBoost · M-Pesa Daraja · Stripe  
**Architecture:** Fully self-hosted, no external AI APIs

---

## 1. Project Overview

PreBuyPc is a paid laptop recommendation service. A user describes their budget and use case through a multi-step form. The backend queries a scraped laptop database, runs the input through a locally-trained XGBoost ML model, and returns ranked recommendations. Results are locked behind a payment gate (M-Pesa primary, Stripe secondary).

**Core user flow:**
```
Landing page
    → Multi-step recommendation form
        → Blurred preview of results + payment prompt
            → Payment (M-Pesa / Stripe)
                → Full recommendation results unlocked
```

---

## 2. Project Structure

```
prebuypc/
├── manage.py
├── requirements.txt
├── .env                        ← environment variables (never commit)
├── prebuypc/
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
│
├── apps/
│   ├── laptops/                ← laptop database models + admin
│   │   ├── models.py
│   │   ├── admin.py
│   │   └── management/
│   │       └── commands/
│   │           └── import_laptops.py   ← CSV/JSON import command
│   │
│   ├── recommender/            ← ML engine + form + results
│   │   ├── models.py           ← UserQuery, Recommendation
│   │   ├── views.py
│   │   ├── forms.py
│   │   ├── urls.py
│   │   ├── engine/
│   │   │   ├── features.py     ← input → numeric feature vector
│   │   │   ├── scorer.py       ← loads model, scores candidates
│   │   │   └── explainer.py    ← generates plain-English explanation
│   │   └── ml/
│   │       ├── model.pkl       ← trained XGBoost model (generated separately)
│   │       ├── label_encoder.pkl
│   │       └── benchmark_scores.json  ← CPU/GPU name → score lookup
│   │
│   ├── payments/               ← M-Pesa + Stripe integration
│   │   ├── models.py           ← Payment
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── mpesa.py            ← Daraja API wrapper
│   │   └── stripe_handler.py
│   │
│   └── accounts/               ← optional user auth (for order history)
│       ├── models.py
│       ├── views.py
│       └── urls.py
│
├── templates/
│   ├── base.html
│   ├── landing.html
│   ├── recommender/
│   │   ├── form_step1.html     ← budget + use case
│   │   ├── form_step2.html     ← preferences
│   │   ├── preview.html        ← blurred results + pay CTA
│   │   └── results.html        ← full unlocked results
│   └── payments/
│       ├── checkout.html
│       └── success.html
│
└── static/
    ├── css/
    ├── js/
    └── img/
```

---

## 3. Database Models

### 3.1 `laptops` app — `models.py`

```python
from django.db import models

class Laptop(models.Model):
    # Identity
    brand = models.CharField(max_length=100)
    model_name = models.CharField(max_length=200)
    
    # Processor
    processor_name = models.CharField(max_length=200)       # raw name e.g. "Intel Core i5-1235U"
    processor_score = models.FloatField(default=0)          # looked up from benchmark_scores.json
    processor_generation = models.IntegerField(default=0)   # e.g. 12 for 12th gen Intel
    
    # Memory & Storage
    ram_gb = models.IntegerField()
    storage_gb = models.IntegerField()
    storage_type = models.CharField(
        max_length=10,
        choices=[("SSD", "SSD"), ("HDD", "HDD"), ("eMMC", "eMMC")]
    )
    
    # GPU
    gpu_name = models.CharField(max_length=200)
    gpu_type = models.CharField(
        max_length=20,
        choices=[("integrated", "Integrated"), ("dedicated", "Dedicated")]
    )
    gpu_score = models.FloatField(default=0)                # looked up from benchmark_scores.json
    
    # Display
    display_size_inches = models.FloatField()
    display_resolution = models.CharField(max_length=20)    # e.g. "1920x1080"
    refresh_rate_hz = models.IntegerField(default=60)
    
    # Physical
    battery_wh = models.FloatField(null=True, blank=True)
    weight_kg = models.FloatField(null=True, blank=True)
    
    # Pricing
    price_kes = models.IntegerField()
    price_usd = models.IntegerField(null=True, blank=True)
    
    # Source
    store_name = models.CharField(max_length=100)
    store_url = models.URLField()
    scraped_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)           # set False for out-of-stock
    
    class Meta:
        ordering = ['price_kes']
    
    def __str__(self):
        return f"{self.brand} {self.model_name} — KES {self.price_kes:,}"
```

---

### 3.2 `recommender` app — `models.py`

```python
import uuid
from django.db import models

class UserQuery(models.Model):
    """Stores each recommendation request. Used for analytics + linking to payment."""
    
    USE_CASE_CHOICES = [
        ("student_general", "General Student Use"),
        ("engineering_cad", "Engineering / CAD"),
        ("architecture", "Architecture / 3D Rendering"),
        ("video_editing", "Video Editing"),
        ("graphic_design", "Graphic Design"),
        ("light_gaming", "Light Gaming"),
        ("competitive_gaming", "Competitive Gaming"),
        ("programming", "Programming / Development"),
        ("office_work", "Office / Business"),
    ]
    
    PORTABILITY_CHOICES = [
        ("high", "Very important — I carry it daily"),
        ("medium", "Somewhat important"),
        ("low", "Not important — mostly desktop use"),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    budget_kes = models.IntegerField()
    primary_use_case = models.CharField(max_length=50, choices=USE_CASE_CHOICES)
    secondary_use_case = models.CharField(max_length=50, choices=USE_CASE_CHOICES, blank=True)
    portability_priority = models.CharField(max_length=10, choices=PORTABILITY_CHOICES)
    preferred_brand = models.CharField(max_length=100, blank=True)  # "" means no preference
    min_ram_gb = models.IntegerField(default=8)
    needs_dedicated_gpu = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Payment status — results locked until this is True
    is_paid = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Query {self.id} | {self.primary_use_case} | KES {self.budget_kes:,}"


class Recommendation(models.Model):
    """The actual laptop picks for a query. Created after ML scoring."""
    query = models.ForeignKey(UserQuery, on_delete=models.CASCADE, related_name='recommendations')
    laptop = models.ForeignKey('laptops.Laptop', on_delete=models.CASCADE)
    rank = models.IntegerField()            # 1 = best pick, 2 = runner-up, 3 = budget pick
    score = models.FloatField()             # raw ML confidence score
    explanation = models.TextField()        # plain-English explanation (from explainer.py)
    label = models.CharField(max_length=50) # e.g. "Best Overall", "Best Value", "Most Portable"
    
    class Meta:
        ordering = ['rank']
```

---

### 3.3 `payments` app — `models.py`

```python
from django.db import models
from apps.recommender.models import UserQuery

class Payment(models.Model):
    PROVIDER_CHOICES = [("mpesa", "M-Pesa"), ("stripe", "Stripe")]
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    
    query = models.OneToOneField(UserQuery, on_delete=models.CASCADE, related_name='payment')
    provider = models.CharField(max_length=10, choices=PROVIDER_CHOICES)
    amount_kes = models.IntegerField()
    
    # M-Pesa specific
    mpesa_checkout_request_id = models.CharField(max_length=100, blank=True)
    mpesa_transaction_id = models.CharField(max_length=100, blank=True)
    phone_number = models.CharField(max_length=15, blank=True)  # e.g. 2547XXXXXXXX
    
    # Stripe specific
    stripe_session_id = models.CharField(max_length=200, blank=True)
    stripe_payment_intent = models.CharField(max_length=200, blank=True)
    
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.provider} | {self.status} | KES {self.amount_kes} | Query {self.query_id}"
```

---

## 4. Recommendation Engine

### 4.1 `engine/features.py` — Feature Engineering

Converts a `UserQuery` object + a `Laptop` queryset into a numeric matrix the ML model can score.

```python
import numpy as np
import json
import os

# Load benchmark scores once at module level
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), '../ml/benchmark_scores.json')
with open(BENCHMARK_PATH) as f:
    BENCHMARKS = json.load(f)

# Use case one-hot encoding order — must match training order exactly
USE_CASE_ORDER = [
    "student_general", "engineering_cad", "architecture",
    "video_editing", "graphic_design", "light_gaming",
    "competitive_gaming", "programming", "office_work"
]

def encode_use_case(use_case_str):
    vec = [0] * len(USE_CASE_ORDER)
    if use_case_str in USE_CASE_ORDER:
        vec[USE_CASE_ORDER.index(use_case_str)] = 1
    return vec

def portability_to_score(portability):
    return {"high": 3, "medium": 2, "low": 1}.get(portability, 2)

def build_feature_matrix(query, candidates):
    """
    Returns:
        - np.ndarray of shape (n_candidates, n_features)
        - list of laptop ids in same order
    """
    rows = []
    ids = []
    
    for laptop in candidates:
        # Budget fit: how well does price fit budget (1.0 = perfect, <1 = over budget)
        budget_fit = query.budget_kes / max(laptop.price_kes, 1)
        budget_fit = min(budget_fit, 1.5)  # cap at 1.5 to avoid outlier dominance
        
        # Portability compatibility
        weight_score = 0
        if laptop.weight_kg:
            if laptop.weight_kg < 1.5:
                weight_score = 3
            elif laptop.weight_kg < 2.0:
                weight_score = 2
            else:
                weight_score = 1
        user_portability = portability_to_score(query.portability_priority)
        portability_match = 1 if weight_score >= user_portability else 0
        
        # Use case encoding
        primary_vec = encode_use_case(query.primary_use_case)
        secondary_vec = encode_use_case(query.secondary_use_case) if query.secondary_use_case else [0]*len(USE_CASE_ORDER)
        
        row = [
            # Hardware specs (normalized where useful)
            laptop.processor_score / 10.0,
            laptop.ram_gb / 64.0,
            laptop.storage_gb / 2000.0,
            1 if laptop.storage_type == "SSD" else 0,
            laptop.gpu_score / 10.0,
            1 if laptop.gpu_type == "dedicated" else 0,
            laptop.display_size_inches / 17.0,
            laptop.refresh_rate_hz / 144.0,
            (laptop.battery_wh or 0) / 100.0,
            
            # Budget & portability
            budget_fit,
            portability_match,
            
            # User preferences
            1 if (not query.preferred_brand or query.preferred_brand.lower() in laptop.brand.lower()) else 0,
            1 if query.needs_dedicated_gpu == (laptop.gpu_type == "dedicated") else 0,
            
            # Use case vectors (primary + secondary)
            *primary_vec,
            *secondary_vec,
        ]
        
        rows.append(row)
        ids.append(laptop.id)
    
    return np.array(rows, dtype=np.float32), ids
```

---

### 4.2 `engine/scorer.py` — ML Model Loader & Scorer

```python
import pickle
import os
import numpy as np
from .features import build_feature_matrix
from apps.laptops.models import Laptop

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/model.pkl')

# Load model once when Django starts (not on every request)
_model = None

def get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model

def get_recommendations(query, top_n=3):
    """
    Takes a UserQuery object.
    Returns a list of dicts: [{'laptop': <Laptop>, 'score': float, 'rank': int}]
    """
    # 1. Filter candidates from DB (within budget ±20%, active listings)
    budget_floor = query.budget_kes * 0.5
    budget_ceil = query.budget_kes * 1.2
    
    candidates = Laptop.objects.filter(
        price_kes__gte=budget_floor,
        price_kes__lte=budget_ceil,
        is_active=True
    )
    
    # Apply hard filters
    if query.needs_dedicated_gpu:
        candidates = candidates.filter(gpu_type="dedicated")
    if query.min_ram_gb:
        candidates = candidates.filter(ram_gb__gte=query.min_ram_gb)
    if query.preferred_brand:
        candidates = candidates.filter(brand__icontains=query.preferred_brand)
    
    candidates = list(candidates)
    
    if len(candidates) < 3:
        # Relax brand filter if too few results
        candidates = list(Laptop.objects.filter(
            price_kes__gte=budget_floor,
            price_kes__lte=budget_ceil,
            is_active=True
        ))
    
    if not candidates:
        return []
    
    # 2. Build feature matrix
    feature_matrix, laptop_ids = build_feature_matrix(query, candidates)
    
    # 3. Score with ML model
    model = get_model()
    scores = model.predict_proba(feature_matrix)[:, 1]  # probability of class "good_match"
    
    # 4. Rank and return top N
    ranked = sorted(zip(laptop_ids, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    
    laptop_map = {l.id: l for l in candidates}
    
    return [
        {
            'laptop': laptop_map[lid],
            'score': float(score),
            'rank': i + 1
        }
        for i, (lid, score) in enumerate(top)
    ]
```

---

### 4.3 `engine/explainer.py` — Template-Based Explanation Generator

```python
def explain(laptop, query, rank):
    lines = []
    use_case = query.primary_use_case
    
    # RAM commentary
    if laptop.ram_gb >= 32:
        lines.append(f"With {laptop.ram_gb}GB of RAM, it handles heavy multitasking and large project files without breaking a sweat.")
    elif laptop.ram_gb >= 16:
        lines.append(f"The {laptop.ram_gb}GB RAM is well-suited for {use_case.replace('_', ' ')} workflows.")
    elif laptop.ram_gb == 8:
        lines.append(f"8GB of RAM covers everyday tasks, though demanding software may slow it down over time.")
    
    # GPU commentary
    if laptop.gpu_type == "dedicated":
        lines.append(f"Its dedicated {laptop.gpu_name} handles graphics-intensive tasks like rendering and visual work.")
    elif use_case in ["video_editing", "graphic_design", "architecture", "competitive_gaming"]:
        lines.append(f"Note: this uses integrated graphics, which may limit performance in {use_case.replace('_', ' ')} applications.")
    
    # Storage
    if laptop.storage_type == "SSD":
        lines.append(f"The {laptop.storage_gb}GB SSD keeps boot times and file transfers fast.")
    else:
        lines.append(f"The HDD storage is spacious but slower — consider an SSD upgrade if speed matters.")
    
    # Processor
    lines.append(f"Powered by the {laptop.processor_name}, it delivers reliable performance for your workload.")
    
    # Budget fit
    savings = query.budget_kes - laptop.price_kes
    if savings > 5000:
        lines.append(f"At KES {laptop.price_kes:,}, it comes in KES {savings:,} under your budget.")
    elif savings < 0:
        lines.append(f"At KES {laptop.price_kes:,}, it's slightly above your budget but worth the stretch for the specs.")
    
    # Portability
    if query.portability_priority == "high" and laptop.weight_kg and laptop.weight_kg < 1.6:
        lines.append(f"Weighing just {laptop.weight_kg}kg, it's easy to carry between classes or offices.")
    
    return " ".join(lines)


RANK_LABELS = {
    1: "Best Overall Pick",
    2: "Strong Runner-Up",
    3: "Best Value Option",
}
```

---

## 5. Views & URL Routing

### 5.1 `recommender/views.py`

```python
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .models import UserQuery, Recommendation
from .forms import RecommendationFormStep1, RecommendationFormStep2
from .engine.scorer import get_recommendations
from .engine.explainer import explain, RANK_LABELS
from payments.utils import calculate_fee

def form_step1(request):
    """Step 1: Budget + primary use case"""
    if request.method == "POST":
        form = RecommendationFormStep1(request.POST)
        if form.is_valid():
            request.session['form_step1'] = form.cleaned_data
            return redirect('recommender:form_step2')
    else:
        form = RecommendationFormStep1()
    return render(request, 'recommender/form_step1.html', {'form': form})


def form_step2(request):
    """Step 2: Preferences"""
    if 'form_step1' not in request.session:
        return redirect('recommender:form_step1')
    
    if request.method == "POST":
        form = RecommendationFormStep2(request.POST)
        if form.is_valid():
            # Merge both steps and create UserQuery
            step1 = request.session.pop('form_step1')
            data = {**step1, **form.cleaned_data}
            
            query = UserQuery.objects.create(**data)
            
            # Run ML engine immediately
            results = get_recommendations(query, top_n=3)
            
            for result in results:
                Recommendation.objects.create(
                    query=query,
                    laptop=result['laptop'],
                    rank=result['rank'],
                    score=result['score'],
                    explanation=explain(result['laptop'], query, result['rank']),
                    label=RANK_LABELS.get(result['rank'], "Pick"),
                )
            
            # Store query ID in session
            request.session['query_id'] = str(query.id)
            return redirect('recommender:preview')
    else:
        form = RecommendationFormStep2()
    return render(request, 'recommender/form_step2.html', {'form': form})


def preview(request):
    """Show blurred teaser. Real results are behind payment gate."""
    query_id = request.session.get('query_id')
    if not query_id:
        return redirect('recommender:form_step1')
    
    query = get_object_or_404(UserQuery, id=query_id)
    
    if query.is_paid:
        return redirect('recommender:results')
    
    fee = calculate_fee(query.budget_kes)
    return render(request, 'recommender/preview.html', {
        'query': query,
        'fee': fee,
    })


def results(request):
    """Full results — only accessible after payment confirmed."""
    query_id = request.session.get('query_id')
    if not query_id:
        return redirect('recommender:form_step1')
    
    query = get_object_or_404(UserQuery, id=query_id, is_paid=True)
    recommendations = query.recommendations.select_related('laptop').order_by('rank')
    
    return render(request, 'recommender/results.html', {
        'query': query,
        'recommendations': recommendations,
    })
```

---

### 5.2 `recommender/forms.py`

```python
from django import forms
from .models import UserQuery

class RecommendationFormStep1(forms.Form):
    budget_kes = forms.IntegerField(
        min_value=20000, max_value=500000,
        label="Your budget (KES)",
        widget=forms.NumberInput(attrs={'placeholder': 'e.g. 80000'})
    )
    primary_use_case = forms.ChoiceField(
        choices=UserQuery.USE_CASE_CHOICES,
        label="What will you mainly use it for?"
    )
    secondary_use_case = forms.ChoiceField(
        choices=[("", "None")] + UserQuery.USE_CASE_CHOICES,
        required=False,
        label="Any secondary use? (optional)"
    )


class RecommendationFormStep2(forms.Form):
    portability_priority = forms.ChoiceField(
        choices=UserQuery.PORTABILITY_CHOICES,
        label="How important is portability / light weight?"
    )
    preferred_brand = forms.CharField(
        required=False,
        label="Preferred brand? (optional)",
        widget=forms.TextInput(attrs={'placeholder': 'e.g. Dell, HP, Lenovo — leave blank for any'})
    )
    min_ram_gb = forms.ChoiceField(
        choices=[(8, "8GB"), (16, "16GB"), (32, "32GB")],
        label="Minimum RAM",
        initial=8
    )
    needs_dedicated_gpu = forms.BooleanField(
        required=False,
        label="Must have a dedicated graphics card?"
    )
```

---

### 5.3 `recommender/urls.py`

```python
from django.urls import path
from . import views

app_name = 'recommender'

urlpatterns = [
    path('', views.form_step1, name='form_step1'),
    path('preferences/', views.form_step2, name='form_step2'),
    path('preview/', views.preview, name='preview'),
    path('results/', views.results, name='results'),
]
```

---

## 6. Payments

### 6.1 Fee Calculation — `payments/utils.py`

```python
def calculate_fee(budget_kes):
    """Tiered flat fee based on user's stated budget."""
    if budget_kes < 50000:
        return 99
    elif budget_kes < 150000:
        return 199
    else:
        return 349
```

---

### 6.2 M-Pesa Integration — `payments/mpesa.py`

Uses the **Safaricom Daraja 2.0 API** (STK Push / Lipa Na M-Pesa Online).

```python
import requests
import base64
from datetime import datetime
from django.conf import settings

def get_access_token():
    consumer_key = settings.MPESA_CONSUMER_KEY
    consumer_secret = settings.MPESA_CONSUMER_SECRET
    credentials = base64.b64encode(f"{consumer_key}:{consumer_secret}".encode()).decode()
    
    response = requests.get(
        "https://api.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials",
        headers={"Authorization": f"Basic {credentials}"}
    )
    return response.json()["access_token"]


def stk_push(phone_number, amount_kes, query_id):
    """
    Initiates STK push to user's phone.
    phone_number: format 2547XXXXXXXX
    Returns the CheckoutRequestID for polling/callback matching.
    """
    token = get_access_token()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    shortcode = settings.MPESA_SHORTCODE
    passkey = settings.MPESA_PASSKEY
    password = base64.b64encode(f"{shortcode}{passkey}{timestamp}".encode()).decode()
    
    payload = {
        "BusinessShortCode": shortcode,
        "Password": password,
        "Timestamp": timestamp,
        "TransactionType": "CustomerPayBillOnline",
        "Amount": amount_kes,
        "PartyA": phone_number,
        "PartyB": shortcode,
        "PhoneNumber": phone_number,
        "CallBackURL": settings.MPESA_CALLBACK_URL,  # your-domain.com/payments/mpesa/callback/
        "AccountReference": str(query_id)[:12],
        "TransactionDesc": "PreBuyPc Recommendation",
    }
    
    response = requests.post(
        "https://api.safaricom.co.ke/mpesa/stkpush/v1/processrequest",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    return response.json()
```

---

### 6.3 M-Pesa Callback — `payments/views.py` (excerpt)

```python
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Payment
from apps.recommender.models import UserQuery

@csrf_exempt
def mpesa_callback(request):
    """Safaricom posts payment result here. Must be publicly accessible HTTPS URL."""
    data = json.loads(request.body)
    
    stk_callback = data['Body']['stkCallback']
    result_code = stk_callback['ResultCode']
    checkout_request_id = stk_callback['CheckoutRequestID']
    
    try:
        payment = Payment.objects.get(mpesa_checkout_request_id=checkout_request_id)
    except Payment.DoesNotExist:
        return JsonResponse({"ResultCode": 0, "ResultDesc": "Accepted"})
    
    if result_code == 0:
        # Payment successful
        metadata = stk_callback.get('CallbackMetadata', {}).get('Item', [])
        mpesa_receipt = next((i['Value'] for i in metadata if i['Name'] == 'MpesaReceiptNumber'), '')
        
        payment.status = "completed"
        payment.mpesa_transaction_id = mpesa_receipt
        payment.save()
        
        # Unlock results
        payment.query.is_paid = True
        payment.query.save()
    else:
        payment.status = "failed"
        payment.save()
    
    return JsonResponse({"ResultCode": 0, "ResultDesc": "Accepted"})


def initiate_mpesa_payment(request):
    """Called when user submits phone number on checkout page."""
    if request.method == "POST":
        query_id = request.session.get('query_id')
        phone = request.POST.get('phone_number')  # user inputs 07XX or 2547XX
        
        query = get_object_or_404(UserQuery, id=query_id)
        fee = calculate_fee(query.budget_kes)
        
        # Normalize phone to 2547XXXXXXXX format
        if phone.startswith('07') or phone.startswith('01'):
            phone = '254' + phone[1:]
        
        result = stk_push(phone, fee, query_id)
        checkout_id = result.get('CheckoutRequestID')
        
        Payment.objects.create(
            query=query,
            provider="mpesa",
            amount_kes=fee,
            phone_number=phone,
            mpesa_checkout_request_id=checkout_id,
            status="pending"
        )
        
        return render(request, 'payments/waiting.html', {
            'query_id': query_id,
            'phone': phone
        })
```

---

### 6.4 Payment Status Polling — `payments/views.py` (excerpt)

After initiating STK push, the user sees a "Waiting for payment..." screen. Poll this endpoint every 3 seconds via JavaScript:

```python
def payment_status(request, query_id):
    """Returns JSON payment status. Polled by frontend JS."""
    query = get_object_or_404(UserQuery, id=query_id)
    return JsonResponse({
        "is_paid": query.is_paid,
        "redirect_url": "/recommender/results/" if query.is_paid else None
    })
```

Frontend JavaScript:
```javascript
const poll = setInterval(async () => {
    const res = await fetch(`/payments/status/${queryId}/`);
    const data = await res.json();
    if (data.is_paid) {
        clearInterval(poll);
        window.location.href = data.redirect_url;
    }
}, 3000);
```

---

### 6.5 Stripe Integration — `payments/stripe_handler.py`

```python
import stripe
from django.conf import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

def create_checkout_session(query, fee_kes, success_url, cancel_url):
    # Convert KES to USD approximately (or handle KES directly if Stripe supports it)
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "kes",
                "product_data": {"name": "PreBuyPc Laptop Recommendation"},
                "unit_amount": fee_kes * 100,  # Stripe uses cents/smallest unit
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"query_id": str(query.id)},
    )
    return session
```

---

## 7. Django Settings

### `settings/base.py` (key additions)

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party
    'django_extensions',
    
    # Local
    'apps.laptops',
    'apps.recommender',
    'apps.payments',
    'apps.accounts',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DB_NAME'),
        'USER': env('DB_USER'),
        'PASSWORD': env('DB_PASSWORD'),
        'HOST': env('DB_HOST', default='localhost'),
        'PORT': '5432',
    }
}

# Session — used to carry query_id between form steps and results
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 86400  # 24 hours
```

### `.env` file (never commit to git)

```
SECRET_KEY=your-django-secret-key
DEBUG=True
DB_NAME=prebuypc
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost

MPESA_CONSUMER_KEY=...
MPESA_CONSUMER_SECRET=...
MPESA_SHORTCODE=...
MPESA_PASSKEY=...
MPESA_CALLBACK_URL=https://yourdomain.com/payments/mpesa/callback/

STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

---

## 8. requirements.txt

```
Django==5.1
psycopg2-binary==2.9.9
python-decouple==3.8
djangorestframework==3.15

# ML
scikit-learn==1.5
xgboost==2.1
numpy==1.26
pandas==2.2

# Payments
stripe==10.3
requests==2.32

# Dev tools
django-extensions==3.2
```

---

## 9. Admin Panel Configuration

Register all models so data can be managed via Django admin:

```python
# laptops/admin.py
from django.contrib import admin
from .models import Laptop

@admin.register(Laptop)
class LaptopAdmin(admin.ModelAdmin):
    list_display = ['brand', 'model_name', 'ram_gb', 'storage_gb', 'storage_type', 'gpu_type', 'price_kes', 'is_active']
    list_filter = ['brand', 'gpu_type', 'storage_type', 'is_active']
    search_fields = ['brand', 'model_name', 'processor_name']
    list_editable = ['is_active']


# recommender/admin.py
from django.contrib import admin
from .models import UserQuery, Recommendation

@admin.register(UserQuery)
class UserQueryAdmin(admin.ModelAdmin):
    list_display = ['id', 'primary_use_case', 'budget_kes', 'is_paid', 'created_at']
    list_filter = ['is_paid', 'primary_use_case']


# payments/admin.py
from django.contrib import admin
from .models import Payment

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['query', 'provider', 'amount_kes', 'status', 'created_at']
    list_filter = ['provider', 'status']
```

---

## 10. ML Model Training (Separate Script — Run Locally)

This is not part of the Django app itself. Run this script once to generate `model.pkl`, then copy it into `apps/recommender/ml/`.

```python
# train_model.py
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load labeled dataset (CSV with columns matching feature matrix)
df = pd.read_csv("labeled_laptops.csv")

# Features (must match features.py build_feature_matrix output order exactly)
feature_cols = [
    'processor_score_norm', 'ram_norm', 'storage_norm', 'is_ssd',
    'gpu_score_norm', 'is_dedicated_gpu', 'display_size_norm', 'refresh_norm', 'battery_norm',
    'budget_fit', 'portability_match', 'brand_match', 'gpu_pref_match',
    # ... use case one-hot columns (primary + secondary)
]

X = df[feature_cols]
y = df['is_good_match']  # 1 = good recommendation, 0 = not

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
```

---

## 11. Main URL Config — `prebuypc/urls.py`

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.recommender.urls', namespace='recommender')),
    path('payments/', include('apps.payments.urls', namespace='payments')),
    path('accounts/', include('apps.accounts.urls', namespace='accounts')),
]
```

---

## 12. Deployment Notes (DigitalOcean Droplet)

- **Recommended:** Ubuntu 22.04 · 4GB RAM · 2 vCPU · 80GB SSD (~$24/month)
- **Web server:** Nginx + Gunicorn
- **Process manager:** Systemd (for Gunicorn) + Celery (for scheduled scraper runs)
- **SSL:** Certbot (Let's Encrypt — free)
- **M-Pesa callback URL must be HTTPS** — set up SSL before testing payments
- Use `django.contrib.staticfiles` + Nginx to serve static files in production
- Set `DEBUG=False` and `ALLOWED_HOSTS` in `settings/production.py`

---

## 13. Build Order for Developer

| Step | Task | Notes |
|------|------|-------|
| 1 | Set up Django project, install dependencies, connect PostgreSQL | Use `settings/development.py` |
| 2 | Build `laptops` models + admin, import first batch of laptops via CSV | Validate data looks correct in admin |
| 3 | Build `features.py` + `scorer.py` — test with dummy model | Use `sklearn.dummy.DummyClassifier` to test wiring |
| 4 | Build form views (step1, step2, preview, results) | Test full flow end-to-end with dummy data |
| 5 | Train real XGBoost model, copy `model.pkl` to `apps/recommender/ml/` | Replace dummy classifier |
| 6 | Integrate M-Pesa STK push + callback + polling | Test on Safaricom sandbox first |
| 7 | Integrate Stripe for card payments | Test with Stripe test keys |
| 8 | UI design (form, preview with blur effect, results page) | Mobile-first — most users on phones |
| 9 | Deploy to DigitalOcean, set up Nginx + SSL | M-Pesa callback requires live HTTPS |
| 10 | End-to-end payment test on live environment | Test both M-Pesa and Stripe real payments |

---

*Document prepared for developer handoff. All code samples are reference implementations — developer should adapt to project conventions and test thoroughly before production deployment.*
