# Configuration

Configuration files are YAML mappings. Component sections use a `name` field plus adapter-specific parameters. Environment variables in the form `${NAME}` or `${NAME:-fallback}` are expanded at load time.

V3 validates the loaded configuration before building a pipeline. The validator checks required sections, component names, chunk sizing, and basic field types so mistakes fail early with actionable messages.

