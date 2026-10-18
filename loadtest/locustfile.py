"""Locust workload: mostly single predictions, some batches, and health checks."""

from __future__ import annotations

import random

from locust import HttpUser, between, task

REVIEWS = [
    "الخدمة ممتازة والتجربة كانت رائعة",
    "المنتج سيئ جداً ولن أشتريه مرة أخرى",
    "المنتج عادي ويؤدي الغرض دون مميزات واضحة",
    "التوصيل سريع والتغليف ممتاز",
    "الجودة أقل من المتوقع والسعر مرتفع",
    "وصل الطلب في الموعد والمنتج كما هو موضح",
    "التجربة مقبولة لكن يمكن تحسين خدمة العملاء",
    "ممتاز بالنسبة للسعر وأنصح بشرائه",
    "لا أنصح بهذا المنتج بسبب ضعف الجودة",
    "لا جديد، المنتج يؤدي الغرض فقط",
    "خدمة العملاء جيدة لكن الشحن تأخر",
    "جودة ممتازة وسأطلبه مرة أخرى بإذن الله",
]


class SentimentUser(HttpUser):
    wait_time = between(1, 3)

    @task(16)  # 80% of weighted tasks
    def predict_single(self):
        text = random.choice(REVIEWS)
        with self.client.post(
            "/predict",
            json={"texts": [text]},
            name="/predict (single)",
            catch_response=True,
        ) as response:
            self._check_prediction_response(response, 1)

    @task(3)  # 15% of weighted tasks
    def predict_batch(self):
        size = random.randint(2, 8)
        texts = random.sample(REVIEWS, k=min(size, len(REVIEWS)))
        with self.client.post(
            "/predict",
            json={"texts": texts},
            name="/predict (batch)",
            catch_response=True,
        ) as response:
            self._check_prediction_response(response, len(texts))

    @task(1)  # 5% of weighted tasks
    def health_check(self):
        with self.client.get(
            "/health", name="/health", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"health endpoint returned HTTP {response.status_code}"
                )

    @staticmethod
    def _check_prediction_response(response, expected_count: int) -> None:
        if response.status_code != 200:
            response.failure(
                f"prediction endpoint returned HTTP {response.status_code}: {response.text[:200]}"
            )
            return
        try:
            body = response.json()
            predictions = body.get("predictions", [])
            if len(predictions) != expected_count:
                response.failure(
                    f"expected {expected_count} predictions, got {len(predictions)}"
                )
                return
            if any(
                item.get("label") not in {"NEGATIVE", "NEUTRAL", "POSITIVE"}
                for item in predictions
            ):
                response.failure("unexpected sentiment label")
        except (ValueError, TypeError, AttributeError) as exc:
            response.failure(f"invalid JSON prediction response: {exc}")
