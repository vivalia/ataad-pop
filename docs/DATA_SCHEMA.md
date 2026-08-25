# Data schema

The clinical dataset is not distributed. Authorized users must construct a table with one row per operation and the following core fields:

| Field | Meaning |
|---|---|
| `Date_of_surgery` | Parseable operation date used to derive calendar-year environments. |
| `Surgeon` | De-identified categorical surgeon environment. |
| `Missingctimage` | `1` when research CT images or structured CT phenotype are unavailable; these records are outside the high-dimensional CT analysis. |
| `Bentall_mechanic_valve` | Recorded mechanical-valve Bentall indicator. |
| `Bentall_bio_valve` | Recorded biological-valve Bentall indicator. |

The primary historical-decision label is derived as the logical OR of the two Bentall indicators. It is not an outcome and must not be interpreted as optimal treatment.

Candidate preoperative predictors and their POP/domain assignments are listed in [`config/feature_metadata.csv`](../config/feature_metadata.csv). All columns must be numeric or coercible to numeric after institutionally approved cleaning. Procedure, cardiopulmonary-bypass, intraoperative, and postoperative variables are not eligible predictors.

## Required local documentation

Before an independent-center run, investigators should maintain a local, nonpublic data dictionary specifying:

- clinical display name and unit;
- permitted preoperative measurement window;
- multiple-measurement selection rule;
- source system or acquisition method;
- clinical coding definition;
- plausibility and correction rules;
- any mapping from local terminology to the public schema.

Dates, free text, direct identifiers, and hospital-specific record identifiers must not be exported with predictions or committed to this repository.

