"""
risk_labeler.py — Operational Risk Labeling for InsightOps

Replaces the original resolution-time-only risk definition with a
defensible operational-risk labeling strategy based on:
  1. category  (NETWORK → high-risk infrastructure)
  2. subcategory (ERROR vs TRAINING, INSTALLATION, etc.)
  3. content text (severity keywords for nuanced classification)

Design rationale:
  - NETWORK tickets represent infrastructure incidents → HIGH risk.
  - SECURITY tickets are NOT all high-risk. Most are routine "Invalid
    Credentials" or PIV card exemptions. Only those with severity keywords
    (breach, unauthorized, widespread) are HIGH.
  - EMAIL / PIV CARD / TRAINING / INSTALLATION / CONFIGURATION / REQUEST
    tickets are routine → LOW risk.
  - SOFTWARE/ERROR is the largest group (~14k tickets). Split by content
    severity keywords.
  - Routine individual access issues (password reset, single-user credential
    errors) are LOW, even if categorized as ACCESS.
  - Production outages, server crashes, data loss, financial fraud, and
    widespread service unavailability are HIGH regardless of category.

The labeler returns a binary label:
  1 = HIGH operational risk
  0 = LOW operational risk
"""

# Routine access keywords -> LOW (unless severe/widespread exists)
ROUTINE_ACCESS_KEYWORDS = [
    'password reset', 'reset my password', 'reset password', 
    'forgot my password', 'forgot password', 'change my password', 
    'change password', 'piv card exemption', '1-day piv', 
    '1 day piv', 'rsa token', 'temporary rsa', 'unlock account',
    'account is locked', 'account locked', 'need access to',
    'request access', 'license request', 'additional licenses',
    'need a temporary', 'request a one-day', 'request a 1-day'
]

# High severity/widespread keywords
WIDESPREAD_KEYWORDS = [
    'employees', 'users', 'customers', 'all users', 
    'multiple users', 'everyone', 'widespread', 'team'
]

LOGIN_FAIL_KEYWORDS = [
    'unable to login', 'unable to log in', 'unable to log into', 
    'cannot login', 'cannot log in', 'cannot log into', 
    'failed to login', 'failed to log in', 'failed to log into', 
    'login failure', 'login failures', 'login issue', 'login issues', 
    'login problem', 'login problems', 'login is down', 'sso is down', 'sso down'
]

OUTAGE_KEYWORDS = [
    'completely down', 'outage', 'downtime', 'halt', 'crash', 
    'fatal', 'kernel panic', 'service unavailable', 'inaccessible', 
    'unaccessible', 'not responding', 'system is down', 'server is down',
    'application is down', 'portal is down', 'service is down'
]

FINANCIAL_KEYWORDS = [
    'money deducted', 'money was deducted', 'deducted by', 
    'unauthorized deduction', 'unauthorized charge', 'financial loss', 
    'charged incorrectly', 'unauthorized transaction', 'billing discrepancy',
    'unusual charges'
]

SECURITY_BREACH_KEYWORDS = [
    'security breach', 'unauthorized access', 'compromise', 'data breach', 
    'hacked', 'hacker', 'intrusion', 'leak'
]

LOW_RISK_SUBCATEGORIES = {
    'TRAINING',
    'INSTALLATION',
    'CONFIGURATION',
    'REQUEST',
    'ACTIVATION',
    'DEACTIVATION',
    'RENEWAL',
    'REPLACEMENT',
    'UPGRADE',
    'OPTIMIZATION'
}

LOW_RISK_CATEGORIES = {
    'EMAIL',
    'PIV CARD'
}


def label_ticket(category: str, subcategory: str, content: str) -> int:
    """
    Assign an operational risk label to a single ticket.

    Parameters
    ----------
    category : str
        Ticket category (e.g., SOFTWARE, NETWORK, SECURITY, …)
    subcategory : str
        Ticket subcategory (e.g., ERROR, ACCESS, INSTALLATION, …)
    content : str
        Ticket description / content text.

    Returns
    -------
    int
        1 = HIGH operational risk, 0 = LOW operational risk
    """
    cat = (category or "").strip().upper()
    subcat = (subcategory or "").strip().upper()
    text = (content or "").lower().strip()

    # 1. Check basic category rules first
    if cat == 'NETWORK':
        return 1
    if cat in LOW_RISK_CATEGORIES:
        return 0
    if subcat in LOW_RISK_SUBCATEGORIES:
        return 0

    # 2. Check for routine access keywords (can be overridden by severe keywords)
    routine_access = any(ra in text for ra in ROUTINE_ACCESS_KEYWORDS)

    # 3. Check for high severity content indicators
    widespread = any(w in text for w in WIDESPREAD_KEYWORDS)
    login_fail = any(lf in text for lf in LOGIN_FAIL_KEYWORDS)
    outage = any(o in text for o in OUTAGE_KEYWORDS)
    financial_impact = any(f in text for f in FINANCIAL_KEYWORDS)
    security_breach = any(sb in text for sb in SECURITY_BREACH_KEYWORDS)

    # Widespread system downtime checking (e.g. system down, server down)
    down_specific = ('server' in text or 'database' in text or 'system' in text or 'application' in text or 'portal' in text or 'service' in text) and ('down' in text or 'not working' in text or 'fails' in text or 'failure' in text)

    # If routine access request and NO evidence of outage/widespread/security breach -> LOW
    if routine_access and not (outage or widespread or security_breach):
        return 0

    # High severity triggers
    if (widespread and (login_fail or down_specific)) or outage or financial_impact or security_breach:
        return 1

    # Hardware malfunctions are generally high risk (system failures)
    if cat == 'HARDWARE' and subcat == 'MALFUNCTION':
        return 1

    return 0


def label_dataframe(df):
    """
    Apply operational risk labels to an entire DataFrame.

    Expects columns: 'category', 'subcategory', 'content'
    Adds column: 'risk_label'

    Returns the DataFrame with the new 'risk_label' column.
    """
    df = df.copy()
    df["risk_label"] = df.apply(
        lambda row: label_ticket(
            row.get("category", ""),
            row.get("subcategory", ""),
            row.get("content", ""),
        ),
        axis=1,
    )
    return df
