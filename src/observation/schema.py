"""Definicja układu 222-elementowego wektora obserwacji.

Wszystkie zakresy używają standardowej konwencji Pythona:
początek jest włączony, koniec jest wyłączony.

Przykład:
OWN_CARDS = slice(0, 52) oznacza indeksy 0-51.
"""

NUM_PLAYERS = 4
NUM_OPPONENTS = 3
NUM_CARDS = 52
NUM_ACTIONS = 5
NUM_STREETS = 4

STREET_SUMMARY_SIZE = 7
OPPONENT_STATS_SIZE = 6


# ---------------------------------------------------------------------------
# Główne bloki obserwacji
# ---------------------------------------------------------------------------

# 0-51: dwie własne karty zakodowane jako one-hot.
OWN_CARDS = slice(0, 52)

# 52-103: karty wspólne zakodowane jako one-hot.
BOARD_CARDS = slice(52, 104)

# 104-107: pozostałe stacki czterech graczy.
PLAYER_STACKS = slice(104, 108)

# 108-111: wkłady graczy na aktualnej ulicy.
STREET_CONTRIBUTIONS = slice(108, 112)

# 112-115: łączne wkłady graczy w aktualne rozdanie.
HAND_CONTRIBUTIONS = slice(112, 116)

# 116-119: czy gracz nadal uczestniczy w turnieju.
PLAYER_ACTIVE = slice(116, 120)

# 120-123: czy gracz spasował w aktualnym rozdaniu.
PLAYER_FOLDED = slice(120, 124)

# 124-127: czy gracz jest all-in.
PLAYER_ALL_IN = slice(124, 128)

# 128-131: pozycja buttona względem obserwującego agenta.
BUTTON_POSITION = slice(128, 132)

# 132-135: [preflop, flop, turn, river].
STREET = slice(132, 136)

# 136: aktualna pula.
POT_INDEX = 136

# 137: rzeczywista kwota potrzebna do sprawdzenia.
TO_CALL_INDEX = 137

# 138-157: ostatnia akcja każdego z czterech graczy.
# Każdy gracz otrzymuje 5 wartości:
# [fold, check/call, raise-half-pot, raise-pot, all-in].
LAST_ACTIONS = slice(138, 158)

# 158-185: podsumowanie preflopa, flopa, turna i rivera.
# Każda ulica otrzymuje 7 wartości:
# [brak agresora, gracz 0, gracz 1, gracz 2, gracz 3,
#  liczba calli, liczba raise'ów].
STREET_SUMMARIES = slice(158, 186)

# 186-203: statystyki trzech przeciwników.
# Każdy przeciwnik otrzymuje 6 wartości:
# [VPIP, PFR, agresja, fold-to-raise,
#  showdown win rate, liczba obserwowanych rozdań].
OPPONENT_STATS = slice(186, 204)

# 204: pot odds.
POT_ODDS_INDEX = 204

# 205-213: kategoria aktualnego układu.
# [high card, pair, two pair, three of a kind,
#  straight, flush, full house, four of a kind,
#  straight flush].
HAND_CATEGORY = slice(205, 214)

# 214-217: informacje o drawach.
# [flush draw, open-ended straight draw,
#  gutshot, backdoor flush draw].
DRAWS = slice(214, 218)

# 218-221: struktura boardu.
# [paired, connected, two-tone, monotone].
BOARD_TEXTURE = slice(218, 222)

OBSERVATION_SIZE = 222


# ---------------------------------------------------------------------------
# Indeksy wewnątrz mniejszych bloków
# ---------------------------------------------------------------------------

# Akcje w bloku ostatniej akcji gracza.
ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_HALF_POT = 2
ACTION_RAISE_POT = 3
ACTION_ALL_IN = 4

# Ulice.
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3

# Statystyki przeciwnika.
STAT_VPIP = 0
STAT_PFR = 1
STAT_AGGRESSION = 2
STAT_FOLD_TO_RAISE = 3
STAT_SHOWDOWN_WIN_RATE = 4
STAT_OBSERVED_HANDS = 5

# Kategorie układu.
CATEGORY_HIGH_CARD = 0
CATEGORY_PAIR = 1
CATEGORY_TWO_PAIR = 2
CATEGORY_THREE_OF_A_KIND = 3
CATEGORY_STRAIGHT = 4
CATEGORY_FLUSH = 5
CATEGORY_FULL_HOUSE = 6
CATEGORY_FOUR_OF_A_KIND = 7
CATEGORY_STRAIGHT_FLUSH = 8

# Drawy.
DRAW_FLUSH = 0
DRAW_OPEN_ENDED = 1
DRAW_GUTSHOT = 2
DRAW_BACKDOOR_FLUSH = 3

# Struktura boardu.
BOARD_PAIRED = 0
BOARD_CONNECTED = 1
BOARD_TWO_TONE = 2
BOARD_MONOTONE = 3


# ---------------------------------------------------------------------------
# Funkcje zwracające zakresy dla konkretnego gracza lub ulicy
# ---------------------------------------------------------------------------

def _validate_index(index: int, size: int, name: str) -> None:
    if not 0 <= index < size:
        raise ValueError(
            f"{name} musi być w zakresie 0-{size - 1}, otrzymano: {index}"
        )


def last_action_slice(relative_seat: int) -> slice:
    """Zwróć pięć wartości ostatniej akcji wybranego gracza.

    relative_seat:
        0 = obserwujący agent,
        1 = następny gracz zgodnie z ruchem wskazówek zegara,
        2 = kolejny gracz,
        3 = ostatni gracz.
    """
    _validate_index(relative_seat, NUM_PLAYERS, "relative_seat")

    start = LAST_ACTIONS.start + relative_seat * NUM_ACTIONS
    return slice(start, start + NUM_ACTIONS)


def street_summary_slice(street: int) -> slice:
    """Zwróć pełny siedmioelementowy blok wybranej ulicy."""
    _validate_index(street, NUM_STREETS, "street")

    start = STREET_SUMMARIES.start + street * STREET_SUMMARY_SIZE
    return slice(start, start + STREET_SUMMARY_SIZE)


def street_aggressor_slice(street: int) -> slice:
    """Zwróć pięć wartości ostatniego agresora wybranej ulicy."""
    block = street_summary_slice(street)
    return slice(block.start, block.start + 5)


def street_call_count_index(street: int) -> int:
    """Zwróć indeks znormalizowanej liczby calli na ulicy."""
    return street_summary_slice(street).start + 5


def street_raise_count_index(street: int) -> int:
    """Zwróć indeks znormalizowanej liczby raise'ów na ulicy."""
    return street_summary_slice(street).start + 6


def opponent_stats_slice(opponent_index: int) -> slice:
    """Zwróć sześć statystyk wybranego przeciwnika.

    opponent_index:
        0 = przeciwnik w relatywnym slocie 1,
        1 = przeciwnik w relatywnym slocie 2,
        2 = przeciwnik w relatywnym slocie 3.
    """
    _validate_index(opponent_index, NUM_OPPONENTS, "opponent_index")

    start = OPPONENT_STATS.start + opponent_index * OPPONENT_STATS_SIZE
    return slice(start, start + OPPONENT_STATS_SIZE)


# ---------------------------------------------------------------------------
# Automatyczna kontrola poprawności schematu
# ---------------------------------------------------------------------------

SCHEMA_BLOCKS = (
    ("own_cards", OWN_CARDS),
    ("board_cards", BOARD_CARDS),
    ("player_stacks", PLAYER_STACKS),
    ("street_contributions", STREET_CONTRIBUTIONS),
    ("hand_contributions", HAND_CONTRIBUTIONS),
    ("player_active", PLAYER_ACTIVE),
    ("player_folded", PLAYER_FOLDED),
    ("player_all_in", PLAYER_ALL_IN),
    ("button_position", BUTTON_POSITION),
    ("street", STREET),
    ("pot", slice(POT_INDEX, POT_INDEX + 1)),
    ("to_call", slice(TO_CALL_INDEX, TO_CALL_INDEX + 1)),
    ("last_actions", LAST_ACTIONS),
    ("street_summaries", STREET_SUMMARIES),
    ("opponent_stats", OPPONENT_STATS),
    ("pot_odds", slice(POT_ODDS_INDEX, POT_ODDS_INDEX + 1)),
    ("hand_category", HAND_CATEGORY),
    ("draws", DRAWS),
    ("board_texture", BOARD_TEXTURE),
)


def validate_schema() -> None:
    """Sprawdź, czy schemat nie ma luk ani nachodzących zakresów."""
    expected_start = 0

    for name, block in SCHEMA_BLOCKS:
        if block.start != expected_start:
            raise RuntimeError(
                f"Nieprawidłowy początek bloku {name!r}: "
                f"oczekiwano {expected_start}, otrzymano {block.start}"
            )

        if block.stop <= block.start:
            raise RuntimeError(f"Blok {name!r} ma nieprawidłowy rozmiar")

        expected_start = block.stop

    if expected_start != OBSERVATION_SIZE:
        raise RuntimeError(
            f"Schemat kończy się na {expected_start}, "
            f"ale OBSERVATION_SIZE wynosi {OBSERVATION_SIZE}"
        )


validate_schema()