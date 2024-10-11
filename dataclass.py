from dataclasses import dataclass, field
from typing import List, T

class FeatureType:
    pass

class Numeric(FeatureType):
    def __repr__(self):
        return '<numeric feature>'

@dataclass
class Categorial(FeatureType):
    levels: dict[T, int] # category levels

    def __repr__(self):
        return f'<categorial feature(levels=[{", ".join(map(str, self.levels))}])>'

@dataclass
class VectorFeature:
    val: List[T]
    tp: FeatureType
    pad_value: T

@dataclass
class VectorDsFeature:
    val: List[List[T]]
    tp: FeatureType
    pad_value: T

    @classmethod
    def numeric(cls, val: List[List[T]], pad_value: T):
        return cls(val=val, tp=Numeric(), pad_value=pad_value)

    @classmethod
    def categorial(cls, val: List[List[T]], pad_value: T, levels: dict[T, int] | None = None):
        if levels is None:
          # infer levels
            levels = dict()
            for va in val:
                for v in va:
                    if not v in levels:
                      levels[v] = len(levels)

            if not pad_value in levels:
                levels[pad_value] = len(levels)

        return cls(val=val, tp=Categorial(levels), pad_value=pad_value)


    def __getitem__(self, ind: int):
        return VectorFeature(val=self.val[ind],
                             tp=self.tp,
                             pad_value=self.pad_value)
@dataclass
class ScalarFeature:
    val: T
    tp: FeatureType

@dataclass
class ScalarDsFeature:
    val: List[T]
    tp: FeatureType

    @classmethod
    def numeric(cls, val: List[T]):
        return cls(val=val, tp=Numeric())

    @classmethod
    def categorial(cls, val: List[T], levels: dict[T, int] | None = None):
        if levels is None:
            # infer levels
            levels = dict()
            for v in val:
                if not v in levels:
                    levels[v] = len(levels)

        return cls(val=val, tp=Categorial(levels))

    def __getitem__(self, ind: int):
        return ScalarFeature(val=self.val[ind],
                             tp=self.tp)
'''
        SEQUENCE OBJECT
'''

@dataclass
class SeqObj:
    """Class for keeping track of a seq in df"""
    
    seq: str

    scalars: dict[str, ScalarFeature]
    vectors: dict[str, VectorFeature]
    
    is_flank_added = False
    reverse: bool = False
    use_reverse_channel: bool = False
    rev: float = 0.0
    feature_channels: list[str] = field(default_factory=list) # names of scalar and vector features to be included as a channel
    
    @property 
    def seqsize(self):
        return len(self.seq)



    
