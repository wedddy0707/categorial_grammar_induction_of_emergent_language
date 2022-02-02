def s2b(s: str) -> bool:
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if s.lower() not in ('false', 'f', 'no', 'n', '0'):
        raise ValueError(f"Unknown argument {s}")
    return False
