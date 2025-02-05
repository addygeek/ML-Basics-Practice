import curses
import json
import os
import random
from collections import deque
from cryptography.fernet import Fernet
from enum import Enum
from noise import pnoise3
import numpy as np
import simpleaudio as sa
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

#region Constants
MAZE_SIZE = 13
TIME_LOOP_LIMIT = 7
SANITY_THRESHOLDS = [75, 50, 25]
CRYPTO_CHALLENGES = 3
SOUND_PATTERN_LENGTH = 5
#endregion

#region Enums
class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    UP = 4
    DOWN = 5

class TimeState(Enum):
    PRESENT = 0
    PAST = 1
    FUTURE = 2

class Faction(Enum):
    CHRONO = 0
    VOID = 1
    OBSERVER = 2

class PuzzleType(Enum):
    CRYPTO = 0
    SOUND = 1
    LOGIC = 2
#endregion

#region Data Classes
@dataclass
class Entity:
    x: int
    y: int
    z: int
    symbol: str
    name: str
    faction: Faction
    memory: List[str]

@dataclass
class TimelineEvent:
    trigger_time: int
    description: str
    effect: Dict[str, int]

@dataclass
class Puzzle:
    puzzle_type: PuzzleType
    solution: str
    encrypted_data: Optional[bytes] = None
    sound_pattern: Optional[List[float]] = None
#endregion

class ChronoLabyrinth:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.running = True
        self.player = Entity(0, 0, 0, '@', "Player", Faction.CHRONO, [])
        self.sanity = 100
        self.time_state = TimeState.PRESENT
        self.current_loop = 1
        self.seen_events = set()
        self.crypto_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.crypto_key)
        self.sound_answers = {}
        self.timeline_events = []
        self.faction_relations = {f: 0 for f in Faction}
        self.entities = []
        self.maze = np.zeros((MAZE_SIZE, MAZE_SIZE, MAZE_SIZE), dtype=bool)
        self.generate_world()
        self.generate_events()
        self.generate_entities()

    #region World Generation
    def generate_world(self):
        seed = random.randint(0, 10000)
        for z in range(MAZE_SIZE):
            for y in range(MAZE_SIZE):
                for x in range(MAZE_SIZE):
                    self.maze[x, y, z] = pnoise3(x/10, y/10, z/10 + seed, octaves=3) > 0.1
        self.maze[MAZE_SIZE//2, MAZE_SIZE//2, MAZE_SIZE//2] = True  # Center always open

    def generate_entities(self):
        factions = list(Faction)
        for _ in range(10):
            while True:
                x, y, z = (random.randint(0, MAZE_SIZE-1) for _ in range(3))
                if self.maze[x, y, z]:
                    self.entities.append(Entity(
                        x, y, z,
                        random.choice(['?', '§', 'Ω', '‡']),
                        random.choice(['Echo', 'Shade', 'Temporal', 'Watcher']),
                        random.choice(factions),
                        []
                    ))
                    break
    #endregion

    #region Game Systems
    def generate_events(self):
        time_marks = sorted(random.sample(range(1, TIME_LOOP_LIMIT*2), 5))
        for t in time_marks:
            self.timeline_events.append(TimelineEvent(
                t,
                random.choice([
                    "The walls bleed temporal energy",
                    "Voices from multiple timelines converge",
                    "A dimensional rift briefly opens"
                ]),
                {"sanity": random.randint(-20, -5)}
            ))

    def temporal_shift(self, new_time_state: TimeState):
        if self.time_state == new_time_state:
            return

        sanity_cost = abs(self.time_state.value - new_time_state.value) * 10
        if self.sanity - sanity_cost < 0:
            self.add_message("Your mind fractures at the temporal strain!")
            self.sanity = max(0, self.sanity - sanity_cost)
            self.check_sanity_effects()
            return

        self.time_state = new_time_state
        self.sanity -= sanity_cost
        self.alter_world()
        self.add_message(f"Shifted to {new_time_state.name} timeline")

    def alter_world(self):
        if self.time_state == TimeState.PAST:
            self.maze = np.rot90(self.maze, axes=(0, 1))
        elif self.time_state == TimeState.FUTURE:
            self.maze = np.rot90(self.maze, axes=(1, 2))
        self.check_sanity_effects()

    def check_sanity_effects(self):
        if self.sanity <= SANITY_THRESHOLDS[2]:
            self.add_message("Reality warps uncontrollably around you!")
            self.generate_entities()
        elif self.sanity <= SANITY_THRESHOLDS[1]:
            self.temporal_shift(random.choice(list(TimeState)))

    def generate_puzzle(self, puzzle_type: PuzzleType) -> Puzzle:
        if puzzle_type == PuzzleType.CRYPTO:
            msg = f"Secret {random.randint(1000,9999)}".encode()
            return Puzzle(puzzle_type, msg.decode(), self.cipher_suite.encrypt(msg))
        elif puzzle_type == PuzzleType.SOUND:
            pattern = [random.choice([0.25, 0.5, 1.0]) for _ in range(SOUND_PATTERN_LENGTH)]
            return Puzzle(puzzle_type, hashlib.sha256(str(pattern).encode()).hexdigest(), sound_pattern=pattern)
        return Puzzle(puzzle_type, "42")

    def verify_puzzle(self, puzzle: Puzzle, answer: str) -> bool:
        if puzzle.puzzle_type == PuzzleType.CRYPTO:
            return self.cipher_suite.decrypt(puzzle.encrypted_data).decode() == answer
        elif puzzle.puzzle_type == PuzzleType.SOUND:
            return hashlib.sha256(str(puzzle.sound_pattern).encode()).hexdigest() == answer
        return puzzle.solution == answer

    def play_sound_pattern(self, pattern: List[float]):
        fs = 44100
        duration = 0.3
        t = np.linspace(0, duration, int(fs * duration), False)
        for freq in pattern:
            note = np.sin(freq * 440 * 2 * np.pi * t)
            audio = note * (2**15 - 1) / np.max(np.abs(note))
            audio = audio.astype(np.int16)
            play_obj = sa.play_buffer(audio, 1, 2, fs)
            play_obj.wait_done()
    #endregion

    #region Game Loop
    def add_message(self, msg: str):
        self.player.memory.append(f"[Loop {self.current_loop}] {msg}")

    def move_player(self, direction: Direction):
        dx, dy, dz = 0, 0, 0
        if direction == Direction.NORTH: dy = -1
        elif direction == Direction.EAST: dx = 1
        elif direction == Direction.SOUTH: dy = 1
        elif direction == Direction.WEST: dx = -1
        elif direction == Direction.UP: dz = 1
        elif direction == Direction.DOWN: dz = -1

        new_x = self.player.x + dx
        new_y = self.player.y + dy
        new_z = self.player.z + dz

        if 0 <= new_x < MAZE_SIZE and 0 <= new_y < MAZE_SIZE and 0 <= new_z < MAZE_SIZE:
            if self.maze[new_x, new_y, new_z]:
                self.player.x, self.player.y, self.player.z = new_x, new_y, new_z
                self.add_message(f"Moved to ({new_x}, {new_y}, {new_z})")
                self.handle_room_entry()
            else:
                self.add_message("Solid wall blocks your path")
        else:
            self.add_message("You reach the edge of reality")

    def handle_room_entry(self):
        current_pos = (self.player.x, self.player.y, self.player.z)
        for e in self.entities:
            if (e.x, e.y, e.z) == current_pos:
                self.handle_encounter(e)

        if random.random() < 0.3:
            self.handle_random_event()

    def handle_encounter(self, entity: Entity):
        self.add_message(f"Encountered {entity.name} ({entity.faction.name})")
        if entity.faction == Faction.VOID:
            self.sanity -= 15
            self.add_message("The Void entity drains your sanity!")
        elif entity.faction == Faction.OBSERVER:
            puzzle = self.generate_puzzle(PuzzleType.LOGIC)
            self.add_message(f"Observer asks: 'What is the answer to everything?'")
            # In real implementation, get user input
            if self.verify_puzzle(puzzle, "42"):
                self.add_message("Observer nods approvingly")
                self.faction_relations[Faction.OBSERVER] += 10
            else:
                self.add_message("Observer frowns and vanishes")
                self.entities.remove(entity)

    def handle_random_event(self):
        event = random.choice(self.timeline_events)
        if event.trigger_time <= self.current_loop and event not in self.seen_events:
            self.seen_events.add(event)
            self.add_message(event.description)
            self.sanity += event.effect.get("sanity", 0)

    def check_win_condition(self) -> bool:
        center = (MAZE_SIZE//2, MAZE_SIZE//2, MAZE_SIZE//2)
        return (self.player.x, self.player.y, self.player.z) == center and self.current_loop <= TIME_LOOP_LIMIT

    def save_game(self, filename: str):
        data = {
            "player": vars(self.player),
            "sanity": self.sanity,
            "current_loop": self.current_loop,
            "seen_events": [vars(e) for e in self.seen_events],
            "faction_relations": {f.name: v for f, v in self.faction_relations.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_game(self, filename: str):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.player = Entity(**data["player"])
                self.sanity = data["sanity"]
                self.current_loop = data["current_loop"]
                self.seen_events = {TimelineEvent(**e) for e in data["seen_events"]}
                self.faction_relations = {Faction[f]: v for f, v in data["faction_relations"].items()}
        except Exception as e:
            self.add_message(f"Load failed: {str(e)}")
    #endregion

    #region UI
    def draw(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        
        # Draw minimap
        map_size = min(MAZE_SIZE, h-10)
        for y in range(map_size):
            for x in range(map_size):
                if self.maze[x, y, self.player.z]:
                    self.stdscr.addch(y, x, '.' if (x, y) != (self.player.x, self.player.y) else '@')
        
        # Draw HUD
        hud = f"Sanity: {self.sanity}% | Loop: {self.current_loop}/{TIME_LOOP_LIMIT} | Time: {self.time_state.name}"
        self.stdscr.addstr(h-5, 0, hud)
        
        # Draw messages
        for i, msg in enumerate(self.player.memory[-4:]):
            self.stdscr.addstr(h-4 + i, 0, msg)
        
        self.stdscr.refresh()

    def input(self) -> str:
        key = self.stdscr.getkey().upper()
        if key == 'Q': 
            self.running = False
        elif key == 'S':
            self.save_game("chrono_save.json")
        elif key == 'L':
            self.load_game("chrono_save.json")
        return key
    #endregion

def main(stdscr):
    curses.curs_set(0)
    game = ChronoLabyrinth(stdscr)
    
    while game.running:
        game.draw()
        key = game.input()
        
        if key == 'KEY_UP': game.move_player(Direction.NORTH)
        elif key == 'KEY_DOWN': game.move_player(Direction.SOUTH)
        elif key == 'KEY_LEFT': game.move_player(Direction.WEST)
        elif key == 'KEY_RIGHT': game.move_player(Direction.EAST)
        elif key == 'U': game.move_player(Direction.UP)
        elif key == 'D': game.move_player(Direction.DOWN)
        elif key == 'T': game.temporal_shift(TimeState((game.time_state.value + 1) % 3))
        
        if game.check_win_condition():
            curses.beep()
            game.add_message("You reach the Chrono Core! Reality stabilizes...")
            game.running = False

if __name__ == "__main__":
    curses.wrapper(main)