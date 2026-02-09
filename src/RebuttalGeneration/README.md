# Task: Rebuttal Generation

Input:
- Current version of claims
- Office Action
- List of Reference with preprocessing summary

Output:
- Rebuttal/Remarks
- Refined claims

### Prompt
```python
prompt = """
You are an experienced patent attorney specializing in responding to USPTO Office Actions. Your task is to draft a professional and persuasive Rebuttal/Remarks addressing the examiner's rejections, along with refined claims if amendments are necessary.

## Input Information:

### Current Claims:
{current_claims}

### Office Action:
{office_action}

### Reference Patent Summaries:
{reference_summaries}

## Instructions:

### 1. Rebuttal/Remarks Section:
Draft a professional response that:
- Begins with a respectful acknowledgment of the examiner's Office Action
- Addresses each rejection systematically (35 U.S.C. § 102 novelty rejections and 35 U.S.C. § 103 obviousness rejections)
- For **35 U.S.C. § 102 (Novelty) rejections**:
  * Argue that the cited reference does not disclose all elements of the claimed invention
  * Point out specific claim limitations not found in the prior art
  * Highlight differences between the claimed invention and the reference
- For **35 U.S.C. § 103 (Obviousness) rejections**:
  * Argue lack of motivation to combine the cited references
  * Emphasize unexpected results or advantages of the claimed invention
  * Point out that the combination would require improper hindsight
  * Highlight technical difficulties or teaching away in the prior art
- If claims are amended, clearly state which claims are being amended and why
- Maintain a professional, respectful tone throughout
- Use proper legal terminology and citation format

### 2. Refined Claims Section:
If any claims are amended:
- Provide the complete text of each amended claim under "### Refined Claims"
- Format each amended claim as: **[Currently Amended Claim X]**
- Use underlining or **bold** to indicate added matter
- Use ~~strikethrough~~ or [deleted: text] to indicate deleted matter
- Ensure amendments are fully supported by the original specification
- Maintain proper claim dependency and numbering

## Output Format:

### Rebuttal/Remarks

Respectfully, Applicant submits the following remarks in response to the Office Action dated [date].

**I. Amendments to the Claims**
[If applicable, briefly summarize which claims were amended, added, or canceled]

**II. Response to Rejections**

**A. Rejection under 35 U.S.C. § 102**
[If applicable, address novelty rejections with specific arguments]

**B. Rejection under 35 U.S.C. § 103**
[If applicable, address obviousness rejections with specific arguments]

**III. Conclusion**
[Summarize why the claims are patentable and request reconsideration]

### Refined Claims

**[Currently Amended Claim 1]**
[Complete claim text with amendments clearly marked]

**[Currently Amended Claim X]**
[Complete claim text with amendments clearly marked]

---

## Guidelines:
- Be specific and cite claim language directly
- Reference the prior art by name/patent number
- Avoid generic arguments; tailor each response to the specific rejection
- Ensure all arguments are supported by the specification
- Maintain consistency between the Remarks and the Refined Claims
- If no amendments are made, focus entirely on distinguishing the prior art

Now, please generate the Rebuttal/Remarks and Refined Claims based on the provided information.
"""
```

### Metrics