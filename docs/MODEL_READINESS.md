# Model readiness: two independent tasks

Every insurance category supports two optional **simulated** trained models:
`fraud_suspicion` and `underwriting_risk`. After running `SETUP_WINDOWS.bat`
or `python scripts/prepare_demo_models.py --all`, there are **28 category/task models**.
Models are versioned and deployed independently by task; an existing active
model is never replaced by the demo-provisioning operation.

| Category | Fraud demo model | Underwriting demo model | Sample claim PDF |
|---|---|---|---|
| Motor | Trainable | Trainable | Included |
| Health | Trainable | Trainable | Included |
| Travel | Trainable | Trainable | Included |
| Life | Trainable | Trainable | Included |
| Property | Trainable | Trainable | Included |
| Commercial | Trainable | Trainable | Included |
| Crop | Trainable | Trainable | Included |
| Marine | Trainable | Trainable | — |
| Personal accident | Trainable | Trainable | — |
| Liability | Trainable | Trainable | — |
| Cyber | Trainable | Trainable | — |
| Pet | Trainable | Trainable | — |
| Gadget | Trainable | Trainable | — |
| Workers' compensation | Trainable | Trainable | — |

**Model-input contract**: Fraud expects seven numeric derived fields and requires
at least five. Underwriting requires exactly three valid values:
coverage amount, prior claims and a nonnegative term length derived from valid
policy start/end dates. Both models abstain on significant out-of-distribution
inputs. Separate risk bands are demonstrative and must not be used for pricing,
fraud accusations or claim decisions.

Image quality and hash duplication are technical indicators, not a tampering
classifier. This website has no externally validated model of any type.

## Real data requirements

For each category, obtain consented/licensed records and precise confirmed-outcome labels, e.g. claims finalized by fraud investigation with reliable outcomes. Include legitimate cases, confirmed fraud, unresolved cases (excluded from supervised binary label), varying claim sizes, historical dates, claim types, invoice amounts, prior claims and incident reporting delays. Where legally justified, study documents, image reuse, and cross-document consistency; avoid unconsented sensitive identifiers. Split by unique entity and/or time so the same customer/claim cannot appear on both sides. Track representativeness, calibration, fairness, data drift, false-accusation costs and recall under the target workflow.
