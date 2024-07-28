def parse_range(range_str):
    # a-b,c-d
    ranges = range_str.split(",")
    numbers = set()
    for r in ranges:
        if "-" in r:
            start, end = map(int, r.split("-"))
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(r))
    return numbers