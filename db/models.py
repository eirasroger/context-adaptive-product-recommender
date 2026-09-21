"""SQLAlchemy models for the system of record.

Two halves live here:

* the **registry** -- indicators, categories, contexts, stakeholders and the
  declarations that bind them.  Changing a row here changes model behaviour, so
  registry changes are reviewed, version-bumping events.
* the **corpus** -- products, their indicator values, comparison sets and labels.

Nothing in this module names a product category.  Categories are rows.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    REAL,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names, so Alembic can autogenerate against SQLite,
# which cannot drop or alter an unnamed constraint.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

WILDCARD = "*"
"""Sentinel used where a declaration applies to every category.

SQLite will not enforce uniqueness over a NULL, so a category-general
declaration stores this sentinel rather than NULL in its primary key.
"""


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _strict(*args, **kwargs):
    """Table arguments for a STRICT table, merging any extra constraints."""
    return (*args, {"sqlite_strict": True, **kwargs})


# ---------------------------------------------------------------------------
# Slot allocation
# ---------------------------------------------------------------------------


class EmbeddingSlot(Base):
    """Append-only integer identity for every entity the model embeds.

    A slot is allocated once and never reused or renumbered.  This is what lets
    a new indicator, context or stakeholder be added as a row without shifting
    any existing weight: the embedding table grows at the tail.
    """

    __tablename__ = "embedding_slot"
    __table_args__ = _strict(UniqueConstraint("table_name", "slot", name="uq_slot"))

    table_name: Mapped[str] = mapped_column(Text, primary_key=True)
    entity_key: Mapped[str] = mapped_column(Text, primary_key=True)
    slot: Mapped[int] = mapped_column(Integer, nullable=False)
    allocated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Registry: indicators
# ---------------------------------------------------------------------------


class IndicatorFamily(Base):
    """A group of indicators that behave alike under a context.

    Family is registry metadata *and* a model input: an indicator's identity
    embedding is composed as ``family + indicator_specific``, so a
    never-before-seen indicator starts from its family's learned prior rather
    than from noise.
    """

    __tablename__ = "indicator_family"
    __table_args__ = _strict()

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    indicators: Mapped[list["Indicator"]] = relationship(back_populates="family")


class Indicator(Base):
    """A measurable property, shared across every category that declares it.

    ``gwp`` is literally the same entity everywhere it appears, so a category
    that reuses it costs zero new parameters.
    """

    __tablename__ = "indicator"
    __table_args__ = _strict(
        CheckConstraint(
            "value_type IN ('continuous','ordinal','nominal','boolean')",
            name="value_type",
        ),
        CheckConstraint("default_direction IN (-1,0,1)", name="direction"),
        # A nominal declaration is expensive -- no direction can be declared, so
        # the model must learn preference from labelled data alone.  Requiring a
        # written justification is how "prefer ordinal wherever an order exists"
        # is enforced rather than merely advised.
        CheckConstraint(
            "value_type <> 'nominal' OR nominal_justification IS NOT NULL",
            name="nominal_justified",
        ),
        CheckConstraint(
            "value_type <> 'nominal' OR default_direction = 0",
            name="nominal_no_direction",
        ),
        CheckConstraint(
            "unit IS NULL OR value_type = 'continuous'", name="unit_continuous_only"
        ),
    )

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    family_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator_family.key"), nullable=False
    )
    value_type: Mapped[str] = mapped_column(Text, nullable=False)
    unit: Mapped[str | None] = mapped_column(Text)
    #: Human-readable semantics, direction and interpretation.  Required from day
    #: one: text-initialised identity embeddings cannot be added retroactively.
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: Direction when no active context declares one.  0 means "direction is
    #: supplied entirely by context", which is a declaration, not an omission.
    default_direction: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    nominal_justification: Mapped[str | None] = mapped_column(Text)
    is_derived: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    family: Mapped[IndicatorFamily] = relationship(back_populates="indicators")
    levels: Mapped[list["IndicatorLevel"]] = relationship(
        back_populates="indicator", order_by="IndicatorLevel.level_index"
    )


class IndicatorDerivation(Base):
    """A derived indicator as a linear combination of authored ones.

    Deliberately not an expression language: coefficients only, so a derivation
    can be reviewed by reading it and can never execute anything.
    """

    __tablename__ = "indicator_derivation"
    __table_args__ = _strict()

    derived_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator.key"), primary_key=True
    )
    source_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator.key"), primary_key=True
    )
    coefficient: Mapped[float] = mapped_column(REAL, nullable=False)


class IndicatorLevel(Base):
    """One level of an ordinal or nominal indicator."""

    __tablename__ = "indicator_level"
    __table_args__ = _strict(
        UniqueConstraint("indicator_key", "level_index", name="uq_level_index"),
        CheckConstraint(
            "normalised_position IS NULL OR "
            "(normalised_position >= 0.0 AND normalised_position <= 1.0)",
            name="position_unit_interval",
        ),
    )

    indicator_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator.key"), primary_key=True
    )
    level_key: Mapped[str] = mapped_column(Text, primary_key=True)
    level_index: Mapped[int] = mapped_column(Integer, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: Where this level sits on the declared scale.  Scale *geometry*, not
    #: utility -- it is what the model receives, and it exists because ordered
    #: scales are rarely equally spaced.  NULL for nominal indicators.
    normalised_position: Mapped[float | None] = mapped_column(REAL)
    #: "A product at this level should never be selected."  Echoed in the API
    #: response alongside the eligibility precondition; never a model input.
    is_disqualifying: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    indicator: Mapped[Indicator] = relationship(back_populates="levels")


# ---------------------------------------------------------------------------
# Registry: categories
# ---------------------------------------------------------------------------


class FunctionalUnit(Base):
    """The declared basis of comparison for a category.

    Never normalise everything to mass: a heavier product would score better per
    kilogram while being worse in every way that matters.
    """

    __tablename__ = "functional_unit"
    __table_args__ = _strict(
        CheckConstraint(
            "quantity_kind IN ('mass','volume','area','length','count','energy')",
            name="quantity_kind",
        )
    )

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    quantity_kind: Mapped[str] = mapped_column(Text, nullable=False)
    unit: Mapped[str] = mapped_column(Text, nullable=False)
    reference_service_life_years: Mapped[float | None] = mapped_column(REAL)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)


class Category(Base):
    """A product category: a set of registry rows, never code."""

    __tablename__ = "category"
    __table_args__ = _strict(
        # Redundant as a constraint, essential as a composite-FK parent: this is
        # what makes the functional-unit invariant enforceable by the engine.
        UniqueConstraint("key", "functional_unit_key", name="uq_category_fu"),
    )

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)
    functional_unit_key: Mapped[str] = mapped_column(
        Text, ForeignKey("functional_unit.key"), nullable=False
    )
    #: What regulatory prefiltering the model assumes has already happened.
    #: Never modelled, always stated, echoed in every API response.
    eligibility_precondition_text: Mapped[str] = mapped_column(Text, nullable=False)
    default_context_key: Mapped[str] = mapped_column(
        Text, ForeignKey("context.key"), nullable=False
    )
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class CategoryIndicator(Base):
    """Which indicators apply to a category, and how relevance is decided."""

    __tablename__ = "category_indicator"
    __table_args__ = _strict(
        UniqueConstraint("category_key", "indicator_key", name="uq_cat_ind"),
        CheckConstraint(
            "relevance_mode IN ('always','context_declared')", name="relevance_mode"
        ),
        CheckConstraint(
            "direction_override IS NULL OR direction_override IN (-1,0,1)",
            name="direction_override",
        ),
        CheckConstraint("control_mode IN ('sweep','exclude')", name="control_mode"),
        # Excluding an indicator from control generation removes it from the
        # behavioural gate too, so the reason has to be written down rather than
        # left as a silent gap in what is tested.
        CheckConstraint(
            "control_mode <> 'exclude' OR control_note IS NOT NULL",
            name="exclusion_justified",
        ),
    )

    category_key: Mapped[str] = mapped_column(
        Text, ForeignKey("category.key"), primary_key=True
    )
    indicator_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator.key"), primary_key=True
    )
    is_required: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: ``always`` -- the relevance flag is 1 whatever the context.
    #: ``context_declared`` -- relevant only when an active context declares over
    #: this indicator.
    relevance_mode: Mapped[str] = mapped_column(
        Text, nullable=False, default="always"
    )
    direction_override: Mapped[int | None] = mapped_column(Integer)
    #: Whether a control case may vary this indicator on its own.
    #:
    #: ``sweep`` -- the declared direction holds across the whole range, so
    #: holding everything else ideal and moving this one asserts something true.
    #: The behavioural suite gates on it.
    #:
    #: ``exclude`` -- the relationship is not monotone across the range, is
    #: disputed, or is still an open question. Generating a control label would
    #: assert something the registry does not actually claim, so nothing is
    #: generated and nothing is gated. An untested indicator is a known gap, not
    #: a silent pass.
    control_mode: Mapped[str] = mapped_column(Text, nullable=False, default="sweep")
    control_note: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)


class IndicatorReferenceRange(Base):
    """The declared range an indicator is normalised against, per category.

    Declared, never fitted from data.  This is what makes a checkpoint
    independent of whichever dataset file happens to be on disk, and it is why
    the same indicator entity can be used at two different functional units.
    """

    __tablename__ = "indicator_reference_range"
    __table_args__ = _strict(
        ForeignKeyConstraint(
            ["category_key", "indicator_key"],
            ["category_indicator.category_key", "category_indicator.indicator_key"],
            name="fk_range_membership",
        ),
        CheckConstraint("ref_high > ref_low", name="range_ordered"),
        CheckConstraint("scale IN ('linear','log')", name="scale"),
        CheckConstraint("scale <> 'log' OR ref_low > 0", name="log_positive"),
        CheckConstraint(
            "shape IN ('monotone','ideal_point','ideal_band')", name="shape"
        ),
        CheckConstraint(
            "shape <> 'ideal_point' OR ideal_value IS NOT NULL", name="ideal_point"
        ),
        CheckConstraint(
            "shape <> 'ideal_band' OR (ideal_low IS NOT NULL AND ideal_high IS NOT NULL)",
            name="ideal_band",
        ),
    )

    category_key: Mapped[str] = mapped_column(Text, primary_key=True)
    indicator_key: Mapped[str] = mapped_column(Text, primary_key=True)
    ref_low: Mapped[float] = mapped_column(REAL, nullable=False)
    ref_high: Mapped[float] = mapped_column(REAL, nullable=False)
    scale: Mapped[str] = mapped_column(Text, nullable=False, default="linear")
    #: ``monotone`` -- more or less is simply better.
    #: ``ideal_point`` / ``ideal_band`` -- too little and too much are both bad.
    shape: Mapped[str] = mapped_column(Text, nullable=False, default="monotone")
    ideal_value: Mapped[float | None] = mapped_column(REAL)
    ideal_low: Mapped[float | None] = mapped_column(REAL)
    ideal_high: Mapped[float | None] = mapped_column(REAL)
    source_note: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Registry: contexts
# ---------------------------------------------------------------------------


class Context(Base):
    """An application context.  Both a declaration and a learned entity."""

    __tablename__ = "context"
    __table_args__ = _strict()

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: A baseline context is always available for the categories that name it as
    #: their default, even if it declares over nothing they hold.
    is_baseline: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class ContextDeclaration(Base):
    """``(context, indicator, direction, priority)``, category-general by default.

    Because declarations name indicators rather than categories, a context's
    availability for a category is *derived* from whether that category holds the
    indicators the context points at.  No per-category context list is
    maintained by hand.
    """

    __tablename__ = "context_declaration"
    __table_args__ = _strict(
        CheckConstraint("direction IN (-1,0,1)", name="direction"),
        CheckConstraint("priority > 0.0 AND priority <= 1.0", name="priority"),
    )

    context_key: Mapped[str] = mapped_column(
        Text, ForeignKey("context.key"), primary_key=True
    )
    indicator_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator.key"), primary_key=True
    )
    #: ``*`` means every category; a category key here overrides the general row.
    category_key: Mapped[str] = mapped_column(
        Text, primary_key=True, default=WILDCARD
    )
    direction: Mapped[int] = mapped_column(Integer, nullable=False)
    priority: Mapped[float] = mapped_column(REAL, nullable=False)
    #: When set, the context is meaningless for a category that lacks this
    #: indicator, so the context is inert there.
    is_required: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ideal_value: Mapped[float | None] = mapped_column(REAL)
    note: Mapped[str | None] = mapped_column(Text)


class CategoryContextOverride(Base):
    """Escape hatch for a context that derivation makes available but that is
    semantically wrong for a category.

    Derivation covers the mechanical case (the indicator is simply absent).  It
    cannot cover the case where a category happens to hold the right indicators
    but the context means nothing for it.  Those are few, so an explicit,
    justified row is the honest answer.
    """

    __tablename__ = "category_context_override"
    __table_args__ = _strict(
        CheckConstraint("is_applicable IN (0,1)", name="is_applicable")
    )

    category_key: Mapped[str] = mapped_column(
        Text, ForeignKey("category.key"), primary_key=True
    )
    context_key: Mapped[str] = mapped_column(
        Text, ForeignKey("context.key"), primary_key=True
    )
    is_applicable: Mapped[int] = mapped_column(Integer, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Registry: stakeholders
# ---------------------------------------------------------------------------


class Stakeholder(Base):
    """A decision-maker archetype, with a learned embedding rather than a one-hot.

    A ninth archetype is a new row; existing checkpoints survive it.
    """

    __tablename__ = "stakeholder"
    __table_args__ = _strict()

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: For archetypes that stop applying outside their domain.  Recorded from day
    #: one, unused by the model.
    domain_scope_text: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class StakeholderPriority(Base):
    """How much a stakeholder cares about a family or a single indicator.

    Generator input only, never a model input: the control labels are produced
    from this table, so feeding it to the model as a feature would make the
    behavioural suite circular.
    """

    __tablename__ = "stakeholder_priority"
    __table_args__ = _strict(
        CheckConstraint("target_kind IN ('family','indicator')", name="target_kind"),
        CheckConstraint("priority >= 0.0 AND priority <= 1.0", name="priority"),
    )

    stakeholder_key: Mapped[str] = mapped_column(
        Text, ForeignKey("stakeholder.key"), primary_key=True
    )
    target_kind: Mapped[str] = mapped_column(Text, primary_key=True)
    target_key: Mapped[str] = mapped_column(Text, primary_key=True)
    priority: Mapped[float] = mapped_column(REAL, nullable=False)


# ---------------------------------------------------------------------------
# Registry: provenance and releases
# ---------------------------------------------------------------------------


class Provenance(Base):
    """Where a label came from."""

    __tablename__ = "provenance"
    __table_args__ = _strict()

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    definition_text: Mapped[str] = mapped_column(Text, nullable=False)


class CategoryProvenanceWeight(Base):
    """Per-category loss weighting across provenances."""

    __tablename__ = "category_provenance_weight"
    __table_args__ = _strict(CheckConstraint("weight >= 0.0", name="weight"))

    category_key: Mapped[str] = mapped_column(
        Text, ForeignKey("category.key"), primary_key=True
    )
    provenance_key: Mapped[str] = mapped_column(
        Text, ForeignKey("provenance.key"), primary_key=True
    )
    weight: Mapped[float] = mapped_column(REAL, nullable=False)


class RegistryRelease(Base):
    """A frozen, hashed serialisation of the whole registry.

    Registry rows stay mutable; a release pins them.  A checkpoint records the
    release version and the snapshot hash, which together make a training run
    reproducible.
    """

    __tablename__ = "registry_release"
    __table_args__ = _strict()

    version: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    #: JSON, ``{table_name: max_slot}`` -- how wide the embedding tables were.
    max_slots: Mapped[str] = mapped_column(Text, nullable=False)
    #: The full canonical mirror, so any checkpoint's registry is recoverable
    #: from the database alone.
    yaml_blob: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class DataSource(Base):
    """An ingested dataset, with the hash of what was actually read."""

    __tablename__ = "data_source"
    __table_args__ = _strict()

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    citation: Mapped[str] = mapped_column(Text, nullable=False)
    doi: Mapped[str | None] = mapped_column(Text)
    ingested_at: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_version: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)


class Product(Base):
    """One alternative.  Values hang off the product, not off set membership."""

    __tablename__ = "product"
    __table_args__ = _strict(
        UniqueConstraint("id", "category_key", name="uq_product_category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_key: Mapped[str] = mapped_column(
        Text, ForeignKey("category.key"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(
        Text, ForeignKey("data_source.key"), nullable=False
    )
    external_ref: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class IndicatorValue(Base):
    """One row per ``(alternative, indicator)``.

    ``present`` is an explicit column, so "applicable to this category but
    missing for this product" is a stored fact rather than something inferred
    from a type check at load time.

    Relevance is deliberately *not* stored here: it is a function of the active
    context, that is of the query, not of the product.
    """

    __tablename__ = "indicator_value"
    __table_args__ = _strict(
        ForeignKeyConstraint(
            ["indicator_key", "level_key"],
            ["indicator_level.indicator_key", "indicator_level.level_key"],
            name="fk_value_level",
        ),
        CheckConstraint("present IN (0,1)", name="present"),
        CheckConstraint(
            "(present = 1 AND (value_num IS NOT NULL OR level_key IS NOT NULL)) OR "
            "(present = 0 AND value_num IS NULL AND level_key IS NULL)",
            name="present_implies_value",
        ),
        CheckConstraint(
            "value_num IS NULL OR level_key IS NULL", name="one_value_channel"
        ),
    )

    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("product.id"), primary_key=True
    )
    indicator_key: Mapped[str] = mapped_column(
        Text, ForeignKey("indicator.key"), primary_key=True
    )
    present: Mapped[int] = mapped_column(Integer, nullable=False)
    value_num: Mapped[float | None] = mapped_column(REAL)
    level_key: Mapped[str | None] = mapped_column(Text)


class ComparisonSet(Base):
    """One decision: a shortlist of alternatives under a context and stakeholders."""

    __tablename__ = "comparison_set"
    __table_args__ = _strict(
        UniqueConstraint("source_key", "external_id", name="uq_source_external"),
        UniqueConstraint("id", "category_key", name="uq_set_category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_key: Mapped[str] = mapped_column(
        Text, ForeignKey("category.key"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(
        Text, ForeignKey("data_source.key"), nullable=False
    )
    #: A real column, never a string prefix on an identifier.
    provenance_key: Mapped[str] = mapped_column(
        Text, ForeignKey("provenance.key"), nullable=False
    )
    generator_key: Mapped[str | None] = mapped_column(Text)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class ComparisonSetMember(Base):
    """An alternative's place in a comparison set.

    The two composite foreign keys are the whole mechanism behind the
    functional-unit invariant: a member row must name one category that matches
    both its set and its product, so a product from a category with a different
    functional unit cannot be inserted at all.
    """

    __tablename__ = "comparison_set_member"
    __table_args__ = _strict(
        UniqueConstraint("set_id", "position", name="uq_set_position"),
        ForeignKeyConstraint(
            ["set_id", "category_key"],
            ["comparison_set.id", "comparison_set.category_key"],
            name="fk_member_set_category",
        ),
        ForeignKeyConstraint(
            ["product_id", "category_key"],
            ["product.id", "product.category_key"],
            name="fk_member_product_category",
        ),
    )

    set_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_key: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    local_key: Mapped[str] = mapped_column(Text, nullable=False)


class ComparisonSetStakeholder(Base):
    """Many-to-many: a decision can be made under several archetypes at once."""

    __tablename__ = "comparison_set_stakeholder"
    __table_args__ = _strict()

    set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("comparison_set.id"), primary_key=True
    )
    stakeholder_key: Mapped[str] = mapped_column(
        Text, ForeignKey("stakeholder.key"), primary_key=True
    )


class ComparisonSetContext(Base):
    """Many-to-many: a decision can sit in several contexts at once."""

    __tablename__ = "comparison_set_context"
    __table_args__ = _strict()

    set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("comparison_set.id"), primary_key=True
    )
    context_key: Mapped[str] = mapped_column(
        Text, ForeignKey("context.key"), primary_key=True
    )


class LabelSet(Base):
    """One labelling pass over one comparison set.

    Separate from the labels themselves so that several independent labellers
    can score the same set without being averaged at ingest time, and so that
    what the scores *mean* is recorded rather than assumed.
    """

    __tablename__ = "label_set"
    __table_args__ = _strict(
        UniqueConstraint(
            "comparison_set_id",
            "provenance_key",
            "labeller_key",
            name="uq_labelset_identity",
        ),
        CheckConstraint(
            "scale_semantics IN ('within_set_relative','absolute_reference')",
            name="scale_semantics",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    comparison_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("comparison_set.id"), nullable=False
    )
    provenance_key: Mapped[str] = mapped_column(
        Text, ForeignKey("provenance.key"), nullable=False
    )
    labeller_key: Mapped[str | None] = mapped_column(Text)
    #: ``absolute_reference`` -- the score is a function of a declared range.
    #: ``within_set_relative`` -- the score is suitability within this set only.
    #: Recording which is which keeps two different meanings from being averaged
    #: together silently.
    scale_semantics: Mapped[str] = mapped_column(Text, nullable=False)
    method_note: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class Label(Base):
    """A preference score and its confidence for one alternative."""

    __tablename__ = "label"
    __table_args__ = _strict(
        CheckConstraint("pref >= 0.0 AND pref <= 1.0", name="pref_unit_interval"),
        CheckConstraint(
            "conf IS NULL OR (conf >= 0.0 AND conf <= 1.0)", name="conf_unit_interval"
        ),
    )

    label_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("label_set.id"), primary_key=True
    )
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("product.id"), primary_key=True
    )
    pref: Mapped[float] = mapped_column(REAL, nullable=False)
    conf: Mapped[float | None] = mapped_column(REAL)
    rationale: Mapped[str | None] = mapped_column(Text)


class DatasetSplit(Base):
    """A stored train/validation/test assignment.

    Splits are recorded rather than re-randomised per run, so a rerun on a grown
    corpus stays comparable with the run before it.
    """

    __tablename__ = "dataset_split"
    __table_args__ = _strict(
        CheckConstraint("fold IN ('train','val','test')", name="fold")
    )

    comparison_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("comparison_set.id"), primary_key=True
    )
    split_key: Mapped[str] = mapped_column(Text, primary_key=True)
    fold: Mapped[str] = mapped_column(Text, nullable=False)


class Snapshot(Base):
    """An immutable, content-hashed materialisation of the corpus.

    Training never reads the live database; it reads one of these.
    """

    __tablename__ = "snapshot"
    __table_args__ = _strict()

    content_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    registry_version: Mapped[str] = mapped_column(
        Text, ForeignKey("registry_release.version"), nullable=False
    )
    split_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    filter_spec: Mapped[str] = mapped_column(Text, nullable=False)
    row_counts: Mapped[str] = mapped_column(Text, nullable=False)
    parquet_path: Mapped[str] = mapped_column(Text, nullable=False)


Index("ix_indicator_value_indicator", IndicatorValue.indicator_key)
Index("ix_member_product", ComparisonSetMember.product_id)
Index("ix_label_product", Label.product_id)
Index("ix_set_provenance", ComparisonSet.provenance_key, ComparisonSet.category_key)
