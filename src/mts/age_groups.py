AGE_GROUPS = [
    (0, 25),
    (26, 35),
    (36, 45),
    (46, 55),
    (56, 65),
    (66, 99)
]


def age_to_group(age: int) -> int:
    for i, (age_from, age_to) in enumerate(AGE_GROUPS):
        if age_from <= age <= age_to:
            return i
    raise ValueError()
