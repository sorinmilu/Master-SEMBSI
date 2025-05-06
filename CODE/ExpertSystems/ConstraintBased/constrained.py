import itertools

# Definim camerele si sloturile
camere = [1, 2, 3]
sloturi = {
    'M': ['8-10', '10-12'],
    'A': ['12-14', '14-16', '16-18']
}
perioade = ['M', 'A']
persoane = [f'P{i}' for i in range(1, 11)]

# Definirea constrangerilor pentru toate persoanele
def expanded_possible_slots(persoana):
    if persoana == 'P1':
        return [(slot, r) for slot in sloturi['M'] for r in [2, 3]]
    elif persoana == 'P2':
        return [(slot, r) for slot in sloturi['M'] for r in camere]
    elif persoana == 'P3':
        return [(slot, 3) for slot in sloturi['A']]
    elif persoana == 'P4':
        return [(slot, r) for slot in sloturi['A'] for r in [2, 3]]
    elif persoana == 'P5':
        return [(slot, 1) for slot in sloturi['M']]
    elif persoana == 'P6':
        return [(slot, r) for slot in sloturi['A'] for r in camere]
    elif persoana == 'P7':
        return [(slot, 2) for t in perioade for slot in sloturi[t]]
    elif persoana == 'P8':
        return [(slot, 1) for slot in sloturi['A']]
    elif persoana == 'P9':
        return [(slot, r) for slot in sloturi['M'] for r in camere if r != 3]
    elif persoana == 'P10':
        return [(slot, r) for t in perioade for slot in sloturi[t] for r in camere]
    else:
        return [(slot, r) for t in perioade for slot in sloturi[t] for r in camere]

# Construim toate cerintele
cerinte = {p: expanded_possible_slots(p) for p in persoane}

# Verificarea constrangerilor
def is_valid(schedule):
    slots = {p: s for p, s in schedule}
    if slots['P1'][0] == slots['P2'][0]:  # P2 nu poate fi in acelasi timp in cladire cu P1
        return False
    if slots['P1'][0] != slots['P5'][0]:  # P5 trebuie sa fie prezenta in acelasi timp cu P1
        return False
    return True

all_combinations = itertools.product(*(cerinte[p] for p in persoane))
valid_schedule = None

for combo in all_combinations:
    schedule = list(zip(persoane, combo))
    perioade = [(t, r) for _, (t, r) in schedule]
    if len(perioade) != len(set(perioade)):
        continue  # conflict: camera a fost deja rezervata
    if is_valid(schedule):
        valid_schedule = schedule
        break

# Imprima rezultatul
if valid_schedule:
    valid_schedule.sort()
    print(f"{'persoana':<6} {'Time':<7} {'Camera'}")
    print('-' * 20)
    for persoana, (slot, room) in valid_schedule:
        print(f"{persoana:<6} {slot:<7} {room}")
else:
    print("Nu am gasit nici o solutie.")

