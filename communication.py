"""
ARIA Communication — Phase 2
Signal channel + compound lexicon for emergent language.

Individual signals crystallise into named symbols after LEXICON_STABILITY_THRESHOLD uses.
Pairs of signals that co-occur during coordination events crystallise into compound
symbols after COMPOUND_THRESHOLD coordination successes.
"""

import json
import os
from datetime import datetime
from config import (
    N_SIGNALS, LEXICON_LOG_PATH, LEXICON_STABILITY_THRESHOLD,
    UNASSIGNED_SYMBOL, COMPOUND_THRESHOLD, SEQUENCE_THRESHOLD
)

SEQUENCE_SYMBOL_POOL = [
    '[M1]', '[M2]', '[M3]', '[M4]', '[M5]', '[M6]',
    '[M7]', '[M8]', '[M9]', '[M10]', '[M11]', '[M12]',
]

SYMBOL_POOL = [
    '[A]', '[B]', '[C]', '[D]', '[E]', '[F]', '[G]', '[H]',
    '[I]', '[J]', '[K]', '[L]', '[M]', '[N]', '[O]', '[P]'
]

COMPOUND_SYMBOL_POOL = [
    '[AB]', '[AC]', '[AD]', '[BC]', '[BD]', '[CD]',
    '[AE]', '[BE]', '[CE]', '[DE]', '[EF]', '[AF]'
]


class LexiconEntry:
    def __init__(self, signal_idx):
        self.signal_idx      = signal_idx
        self.symbol          = UNASSIGNED_SYMBOL
        self.use_count       = 0
        self.coord_successes = 0
        self.assigned        = False

    def to_dict(self):
        return {
            'signal_idx':      self.signal_idx,
            'symbol':          self.symbol,
            'use_count':       self.use_count,
            'coord_successes': self.coord_successes,
            'assigned':        self.assigned
        }


class CompoundEntry:
    """A crystallised or crystallising compound of two signals."""

    def __init__(self, sig_a, sig_b):
        self.sig_pair        = (min(sig_a, sig_b), max(sig_a, sig_b))
        self.symbol          = UNASSIGNED_SYMBOL
        self.use_count       = 0
        self.coord_successes = 0
        self.crystallised    = False

    def to_dict(self):
        return {
            'sig_pair':        list(self.sig_pair),
            'symbol':          self.symbol,
            'use_count':       self.use_count,
            'coord_successes': self.coord_successes,
            'crystallised':    self.crystallised
        }


class CompoundLexicon:
    """
    Tracks pairs of signals co-sent by different agents in the same step.
    When COMPOUND_THRESHOLD coordination successes are associated with the
    same pair, the pair crystallises into a compound symbol.
    """

    def __init__(self):
        self.entries      = {}   # (sig_a, sig_b) -> CompoundEntry
        self._symbol_pool = list(COMPOUND_SYMBOL_POOL)

    def record_pair(self, sig_a, sig_b, coord_achieved):
        key = (min(sig_a, sig_b), max(sig_a, sig_b))
        if key not in self.entries:
            self.entries[key] = CompoundEntry(*key)
        entry = self.entries[key]
        entry.use_count += 1
        if coord_achieved:
            entry.coord_successes += 1
        if not entry.crystallised and entry.coord_successes >= COMPOUND_THRESHOLD:
            self._crystallise(entry)

    def _crystallise(self, entry):
        if self._symbol_pool:
            entry.symbol      = self._symbol_pool.pop(0)
            entry.crystallised = True

    def get_summary(self):
        return {str(k): v.to_dict() for k, v in self.entries.items()}

    def crystallised_count(self):
        return sum(1 for e in self.entries.values() if e.crystallised)

    def to_list(self):
        return [e.to_dict() for e in self.entries.values()]

    @classmethod
    def from_list(cls, data):
        obj = cls()
        for d in data:
            key = tuple(d['sig_pair'])
            e = CompoundEntry(*key)
            e.symbol          = d['symbol']
            e.use_count       = d['use_count']
            e.coord_successes = d['coord_successes']
            e.crystallised    = d['crystallised']
            obj.entries[key]  = e
            if e.crystallised and e.symbol in obj._symbol_pool:
                obj._symbol_pool.remove(e.symbol)
        return obj

    def inherit_from(self, parent):
        for key, pe in parent.entries.items():
            if pe.crystallised and key not in self.entries:
                e = CompoundEntry(*key)
                e.symbol       = pe.symbol
                e.crystallised = True
                obj_entries_key = key
                self.entries[obj_entries_key] = e
                if e.symbol in self._symbol_pool:
                    self._symbol_pool.remove(e.symbol)


class SequenceEntry:
    """A variable-length message sequence that may crystallise into a named symbol."""

    def __init__(self, sequence):
        self.sequence        = tuple(sequence)
        self.symbol          = UNASSIGNED_SYMBOL
        self.use_count       = 0
        self.coord_successes = 0
        self.crystallised    = False

    def to_dict(self):
        return {
            'sequence':        list(self.sequence),
            'symbol':          self.symbol,
            'use_count':       self.use_count,
            'coord_successes': self.coord_successes,
            'crystallised':    self.crystallised,
        }


class SequenceLexicon:
    """
    Tracks variable-length message sequences. Sequences of 2+ tokens that
    co-occur with coordination events crystallise into named symbols after
    SEQUENCE_THRESHOLD successes.
    """

    def __init__(self):
        self.entries      = {}   # tuple -> SequenceEntry
        self._symbol_pool = list(SEQUENCE_SYMBOL_POOL)

    def record(self, sequence, coord_achieved):
        key = tuple(sequence)
        if len(key) < 2:
            return  # single tokens handled by individual lexicon
        if key not in self.entries:
            self.entries[key] = SequenceEntry(key)
        entry = self.entries[key]
        entry.use_count += 1
        if coord_achieved:
            entry.coord_successes += 1
        if not entry.crystallised and entry.coord_successes >= SEQUENCE_THRESHOLD:
            self._crystallise(entry)

    def _crystallise(self, entry):
        if self._symbol_pool:
            entry.symbol      = self._symbol_pool.pop(0)
            entry.crystallised = True

    def crystallised_count(self):
        return sum(1 for e in self.entries.values() if e.crystallised)

    def get_summary(self):
        return {str(list(k)): v.to_dict() for k, v in self.entries.items()}

    def to_list(self):
        return [e.to_dict() for e in self.entries.values()]

    @classmethod
    def from_list(cls, data):
        obj = cls()
        for d in data:
            key = tuple(d['sequence'])
            e = SequenceEntry(key)
            e.symbol          = d['symbol']
            e.use_count       = d['use_count']
            e.coord_successes = d['coord_successes']
            e.crystallised    = d['crystallised']
            obj.entries[key]  = e
            if e.crystallised and e.symbol in obj._symbol_pool:
                obj._symbol_pool.remove(e.symbol)
        return obj

    def inherit_from(self, parent):
        for key, pe in parent.entries.items():
            if pe.crystallised and key not in self.entries:
                e = SequenceEntry(key)
                e.symbol       = pe.symbol
                e.crystallised = True
                self.entries[key] = e
                if e.symbol in self._symbol_pool:
                    self._symbol_pool.remove(e.symbol)


class CommunicationChannel:
    _BUF_LIMIT = 50   # flush write buffer every N records

    def __init__(self, append_log=False):
        self.lexicon          = {i: LexiconEntry(i) for i in range(N_SIGNALS)}
        self.compound_lexicon = CompoundLexicon()
        self.sequence_lexicon = SequenceLexicon()
        self._symbol_pool     = list(SYMBOL_POOL)
        self.total_signals    = 0
        os.makedirs(os.path.dirname(LEXICON_LOG_PATH), exist_ok=True)
        # Open once for the session — avoids open/close overhead on every signal write
        mode = 'a' if append_log else 'w'
        self._log_fh  = open(LEXICON_LOG_PATH, mode)
        self._log_buf = []
        if not append_log:
            self._write({'event': 'session_start',
                         'timestamp': self._ts(),
                         'n_signals': N_SIGNALS})

    def inherit_from(self, parent_channel):
        for sig_idx, entry in self.lexicon.items():
            parent_entry = parent_channel.lexicon.get(sig_idx)
            if parent_entry and parent_entry.assigned:
                entry.symbol   = parent_entry.symbol
                entry.assigned = True
                if entry.symbol in self._symbol_pool:
                    self._symbol_pool.remove(entry.symbol)

        self.compound_lexicon.inherit_from(parent_channel.compound_lexicon)
        self.sequence_lexicon.inherit_from(parent_channel.sequence_lexicon)

        self._write({
            'event':     'lexicon_inherited',
            'timestamp': self._ts(),
            'symbols':   {str(i): e.symbol for i, e in self.lexicon.items() if e.assigned},
            'compounds': self.compound_lexicon.crystallised_count()
        })

    def record_signal(self, sender_id, signal_idx, state_context, coord_achieved):
        entry = self.lexicon[signal_idx]
        entry.use_count += 1
        if coord_achieved:
            entry.coord_successes += 1
        self.total_signals += 1

        if not entry.assigned and entry.use_count >= LEXICON_STABILITY_THRESHOLD:
            self._assign_symbol(entry)

        self._write({
            'event':          'signal',
            'timestamp':      self._ts(),
            'sender':         sender_id,
            'signal_idx':     signal_idx,
            'symbol':         entry.symbol,
            'use_count':      entry.use_count,
            'coord_achieved': coord_achieved,
            'state_context':  list(state_context)
        })

    def record_message(self, sender_id, message_tokens, coord_achieved):
        """Record a completed variable-length message for sequence crystallisation."""
        if len(message_tokens) >= 2:
            self.sequence_lexicon.record(message_tokens, coord_achieved)
            self._write({
                'event':          'message',
                'timestamp':      self._ts(),
                'sender':         sender_id,
                'sequence':       message_tokens,
                'length':         len(message_tokens),
                'coord_achieved': coord_achieved,
            })

    def record_signal_pair(self, sig_a, sig_b, coord_achieved):
        """Record co-occurrence of two distinct signals for compound crystallisation."""
        if sig_a != sig_b:
            self.compound_lexicon.record_pair(sig_a, sig_b, coord_achieved)

    def _assign_symbol(self, entry):
        if self._symbol_pool:
            entry.symbol   = self._symbol_pool.pop(0)
            entry.assigned = True
            self._write({
                'event':                   'symbol_assigned',
                'timestamp':               self._ts(),
                'signal_idx':              entry.signal_idx,
                'symbol':                  entry.symbol,
                'use_count_at_assignment': entry.use_count
            })

    def get_display_symbol(self, signal_idx):
        entry = self.lexicon[signal_idx]
        return entry.symbol if entry.assigned else f'S{signal_idx}'

    def get_lexicon_summary(self):
        return {i: entry.to_dict() for i, entry in self.lexicon.items()}

    def assigned_count(self):
        return sum(1 for e in self.lexicon.values() if e.assigned)

    def flush_log(self):
        """Write buffered log records to disk. Call before saving or on exit."""
        if self._log_buf:
            self._log_fh.write('\n'.join(self._log_buf) + '\n')
            self._log_buf.clear()
            self._log_fh.flush()

    def __del__(self):
        try:
            self.flush_log()
            self._log_fh.close()
        except Exception:
            pass

    @staticmethod
    def _ts():
        return datetime.now().isoformat()

    def _write(self, record, mode='a'):
        self._log_buf.append(json.dumps(record))
        if len(self._log_buf) >= self._BUF_LIMIT:
            self.flush_log()
