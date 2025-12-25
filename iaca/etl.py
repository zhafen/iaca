"""
ETL workflow for registry processing, based on base_manifest/etl.yaml.
"""

import copy
from typing import List, Dict, Callable
import glob
import os

import networkx as nx
import pandas as pd

from . import data
from .transform import system, preprocess, docs
from .extract import YAMLExtractor, PythonExtractor


# Extraction system: handles reading and parsing entities from YAML
class ExtractSystem:

    def __init__(self):
        self.yaml_extractor = YAMLExtractor()
        self.python_extractor = PythonExtractor()

    def extract_entities(
        self,
        filename_patterns: str | List[str] | Dict[str, List[str]] = [],
        root_dir: str = None,
        input_yaml: str = None,
        system_filename_patterns: List[str] = [
            "../base_manifest/*.yaml",
            "../base_manifest/*.yml",
            "./**/*.py",
        ],
    ) -> data.Registry:

        if root_dir is None:
            root_dir = os.getcwd()

        # Always include base manifest and source files
        source_dir = os.path.dirname(os.path.abspath(__file__))
        # Patterns relative to the source dir.
        system_filename_patterns = [
            os.path.abspath(f"{source_dir}/{pattern}")
            for pattern in system_filename_patterns
        ]

        # Group the filename patterns by source
        filename_patterns_by_source = {
            "system": system_filename_patterns,
        }

        # Parse the filename patterns input,
        # first copying ensure we don't modify the original list
        filename_patterns = copy.deepcopy(filename_patterns)
        if isinstance(filename_patterns, str):
            filename_patterns_by_source["user"] = [filename_patterns]
        elif isinstance(filename_patterns, list):
            filename_patterns_by_source["user"] = filename_patterns
        elif isinstance(filename_patterns, dict):
            filename_patterns_by_source.update(filename_patterns)
        else:
            raise ValueError(
                "filename_patterns must be a string, list of strings, or dict"
            )

        # Iterate over all filename patterns by source
        self.filenames = {}
        entities = []
        for source, filename_patterns in filename_patterns_by_source.items():
            # Resolve paths relative to root
            filename_patterns = [
                (
                    pattern
                    if os.path.isabs(pattern)
                    else os.path.abspath(f"{root_dir}/{pattern}")
                )
                for pattern in filename_patterns
            ]

            # Iterate over the files
            for pattern in filename_patterns:
                filenames = glob.glob(pattern, recursive=True)
                for filename in filenames:
                    self.filenames.setdefault(source, []).append(filename)

                    # Choose extractor based on file type
                    if filename.endswith((".yaml", ".yml")):
                        extractor = YAMLExtractor()
                    elif filename.endswith(".py"):
                        extractor = PythonExtractor()
                    else:
                        continue

                    # Perform extraction
                    entities_i = extractor.extract(filename, root_dir=root_dir)
                    entities += entities_i

                    # Add the source components
                    entity_names = pd.DataFrame(entities_i)["entity"].unique()
                    entities += [
                        {
                            "entity": e,
                            "comp_key": pd.NA,
                            "component_type": "entity_source",
                            "component": {
                                "source": source,
                                "filename_pattern": pattern,
                                "filename": filename,
                            },
                        }
                        for e in entity_names
                    ]

        # Add direct input YAML if provided
        if input_yaml is not None:
            entities_i = self.yaml_extractor.extract_from_input(
                input_yaml, source="input"
            )
            entities += entities_i

            # Add the source components
            entity_names = pd.DataFrame(entities_i)["entity"].unique()
            entities += [
                {
                    "entity": e,
                    "comp_key": pd.NA,
                    "component_type": "entity_source",
                    "component": {
                        "source": "input",
                    },
                }
                for e in entity_names
            ]

        entities = pd.DataFrame(entities)

        return self.load_entities_to_registry(entities)

    def load_entities_to_registry(self, entities: pd.DataFrame) -> data.Registry:

        # Convert to a registry
        registry = data.Registry(
            {
                # We don't want the component_type column in the final registry
                # since it was just used to group the components.
                # We also don't care about the index, so reset it.
                key: df.drop(columns="component_type")
                for key, df in entities.groupby("component_type")
            }
        )

        # We also record the mapping of components to entities in the "compinst"
        # component. We take the time to use the same format as the other components.
        registry["compinst"] = entities[
            ["entity", "comp_key", "component_type"]
        ].set_index(["entity", "comp_key"])

        return registry


# Transform system: handles all transforms on the registry
class TransformSystem:

    def apply_transforms(
        self,
        registry: data.Registry,
        transforms: List[Callable[[data.Registry], data.Registry]],
    ) -> data.Registry:
        # TODO: Delete?
        pass

    def get_transform_order(self, dependencies: Dict[str, List[str]]) -> List[str]:
        # TODO: Delete?
        pass

    def apply_transform(
        self,
        registry: data.Registry,
        transformer,
        components_mapping: dict[str, data.View],
        mode: str = "overwrite",
    ) -> data.Registry:

        # Copy registry to avoid mutation
        new_registry = registry.copy()
        for target_comp, source_view in components_mapping.items():
            # We make a kwargs dictionary so we can easily include the registry
            try:
                result = transformer.transform(
                    registry.resolve_view(source_view),
                    registry,
                )
                new_registry.set(
                    target_comp,
                    result,
                    mode=mode,
                )
            except Exception as e:
                raise ValueError(
                    f"Transformer {transformer} failed to transform component "
                    f"'{target_comp}' with source '{source_view}'"
                ) from e
        return new_registry

    def apply_preprocess_transforms(self, registry: data.Registry) -> data.Registry:

        registry = self.normalize_components(registry)
        registry = self.extract_compdefs(registry)
        registry = self.validate_compdefs(registry)

        return registry

    def normalize_components(self, registry: data.Registry) -> data.Registry:
        """Normalize component schemas in the registry.

        Parameters
        ----------
        registry : data.Registry
            The registry containing the components to normalize.

        Returns
        -------
        data.Registry
            The registry with normalized components.

        Metadata
        --------
        - todo:
            value: >
                Don't hardcode the skip_normalization list here. However, it's difficult
                to put it as a parameter_set because this transform is called early in
                the process, before parameter_sets are extracted.
            priority: 0.1
        """

        # Check raw component definitions for normalize=False
        skip_normalization = []
        if "component" in registry:
            for entity in registry["component"].index.get_level_values("entity").unique():
                try:
                    comp_rows = registry["component"].loc[entity]
                    # Handle both single row and multiple rows
                    if isinstance(comp_rows, pd.Series):
                        comp_rows = pd.DataFrame([comp_rows])
                    
                    for _, row in comp_rows.iterrows():
                        comp_data = row["component"]
                        if isinstance(comp_data, dict) and comp_data.get("normalize") is False:
                            skip_normalization.append(entity)
                            break
                except (KeyError, AttributeError):
                    pass

        return self.apply_transform(
            registry,
            preprocess.ComponentNormalizer(),
            # The components are transformed individually, with a few exceptions.
            components_mapping={
                comp: data.View(comp)
                for comp in registry.keys()
                if comp not in skip_normalization
            },
        )

    def extract_compdefs(self, registry: data.Registry) -> data.Registry:

        return self.apply_transform(
            registry,
            preprocess.ComponentDefExtractor(),
            # We only transform a single component, "component",
            # and it receives a view that joins the "component" and "fields"
            # components.
            components_mapping={
                "compdef": data.View(["component", "fields"], join_how="outer")
            },
        )

    def validate_compdefs(self, registry: data.Registry) -> data.Registry:
        """First we validate the component definitions, since they'll be used
        to validate the components themselves.
        We also do this in postprocessing.
        """

        registry = self.apply_transform(
            registry,
            preprocess.ComponentValidator(),
            components_mapping={"compdef": data.View("compdef")},
        )

        return self.apply_transform(
            registry,
            preprocess.ComponentValidator(),
            components_mapping={
                comp: data.View(comp) for comp in registry.keys() if comp != "compdef"
            },
        )

    def apply_system_transforms(self, registry: data.Registry) -> data.Registry:

        # Parse links components into link components
        registry = self.apply_transform(
            registry,
            system.LinksParser(),
            components_mapping={"link": data.View("links")},
            mode="upsert",
        )

        # Collect components with link_type tags into the link component df
        registry = self.apply_transform(
            registry,
            system.LinkCollector(),
            components_mapping={"link": data.View("link")},
            mode="upsert",
        )

        registry = self.build_graph_from_links(registry)

        # Analyze requirements
        registry = self.apply_transform(
            registry,
            system.RequirementAnalyzer(),
            components_mapping={"requirement": data.View("requirement")},
            mode="overwrite",
        )

        return registry

    def build_graph_from_links(self, registry: data.Registry) -> data.Registry:
        """
        Metadata
        ----------
        - todo: >
            Probably delete this, because it makes more sense to just make graphs as
            needed because the filtering is much more powerful *before* it's turned
            into a graph.
        """

        # Build a graph from the links
        registry.graph = nx.from_pandas_edgelist(
            registry.view("link"),
            source="source",
            target="target",
            edge_key="link_type",
            create_using=nx.MultiDiGraph,
        )
        registry.graph.add_nodes_from(registry.entities)
        registry = self.apply_transform(
            registry,
            system.GraphAnalyzer(),
            components_mapping={"node": data.View("link")},
            mode="overwrite",
        )

        return registry

    def apply_postprocess_transforms(self, registry: data.Registry) -> data.Registry:

        registry = self.validate_compdefs(registry)

        return registry

    def build_docs(self, registry: data.Registry) -> data.Registry:

        # Prepare documents
        registry = self.apply_transform(
            registry,
            docs.DocsGeneratorPreparer(),
            components_mapping={"documentation": data.View("documentation")},
            mode="upsert",
        )

        # Generate documents
        registry = self.apply_transform(
            registry,
            docs.DocsGenerator(),
            components_mapping={"documentation": data.View("documentation")},
            mode="overwrite",
        )

        return registry
