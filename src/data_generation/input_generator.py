"""Input generator for creating diverse synthetic inputs."""

import random
from typing import Dict, Any, List, Callable
from loguru import logger


class InputGenerator:
    """Generator for creating diverse input variations for dataset generation."""

    def __init__(self, openai_client=None):
        """
        Initialize input generator.

        Args:
            openai_client: Optional OpenAI client for AI-powered generation
        """
        self.client = openai_client
        self.generators = {}
        self._register_default_generators()

    def _register_default_generators(self):
        """Register default input generators for common tasks."""
        self.generators["cli_parsing"] = self._generate_cli_input
        self.generators["config_generation"] = self._generate_config_input
        self.generators["troubleshooting"] = self._generate_troubleshooting_input
        self.generators["yang_conversion"] = self._generate_yang_input

    def register_generator(self, task_name: str, generator_func: Callable):
        """
        Register a custom generator for a task.

        Args:
            task_name: Name of the task
            generator_func: Function that generates inputs
        """
        self.generators[task_name] = generator_func
        logger.info(f"Registered custom generator for task: {task_name}")

    def generate_input(self, task_name: str, task_config: Dict[str, Any]) -> Any:
        """
        Generate a synthetic input for a task.

        Args:
            task_name: Name of the task
            task_config: Task configuration from prompts.yaml

        Returns:
            Generated input (format depends on task)
        """
        # Use custom generator if available
        if task_name in self.generators:
            return self.generators[task_name](task_config)

        # Fallback: use AI-powered generation
        if self.client:
            return self._ai_generate_input(task_name, task_config)

        # Last resort: use examples
        return self._example_based_input(task_config)

    def _generate_cli_input(self, task_config: Dict[str, Any]) -> str:
        """Generate synthetic CLI output for parsing tasks."""
        # Interface types
        interface_types = [
            "GigabitEthernet",
            "TenGigabitEthernet",
            "FastEthernet",
            "Ethernet",
            "Loopback",
            "Vlan",
        ]

        # States
        admin_states = ["up", "down", "administratively down"]
        protocol_states = ["up", "down"]

        # Random interface
        iface_type = random.choice(interface_types)
        slot = random.randint(0, 3)
        port = random.randint(0, 48)
        subport = random.randint(0, 4)

        interface_name = f"{iface_type}{slot}/{port}/{subport}"

        admin_state = random.choice(admin_states)
        protocol_state = random.choice(protocol_states) if admin_state == "up" else "down"

        # Build CLI output
        cli_output_templates = [
            # Cisco IOS style
            f"""{interface_name} is {admin_state}, line protocol is {protocol_state}
  Hardware is {iface_type}, address is {self._random_mac()}
  Internet address is {self._random_ip()}/{random.choice([24, 28, 30])}
  MTU {random.choice([1500, 9000, 1492])} bytes, BW {random.choice([1000000, 10000000, 100000])} Kbit
  Encapsulation ARPA, loopback not set
  Last input {random.randint(0, 300)}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}, output {random.randint(0, 300)}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}
  {random.randint(0, 1000000)} packets input, {random.randint(0, 1000000000)} bytes
  {random.randint(0, 1000000)} packets output, {random.randint(0, 1000000000)} bytes""",

            # Shorter version
            f"""{interface_name} is {admin_state}, line protocol is {protocol_state}
  Description: {random.choice(['UPLINK', 'DOWNLINK', 'CUSTOMER_LINK', 'BACKUP'])}
  Internet address is {self._random_ip()}/{random.choice([24, 30])}""",

            # With errors
            f"""{interface_name} is {admin_state}, line protocol is {protocol_state}
  {random.randint(0, 100)} input errors, {random.randint(0, 50)} CRC, {random.randint(0, 20)} frame
  {random.randint(0, 100)} output errors, {random.randint(0, 10)} collisions""",
        ]

        return random.choice(cli_output_templates)

    def _generate_config_input(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic configuration parameters."""
        device_types = ["cisco_ios", "cisco_iosxr", "juniper_junos", "arista_eos"]

        config_params = {
            "device_type": random.choice(device_types),
            "hostname": f"router-{random.randint(1, 999):03d}",
            "interfaces": [],
        }

        # Generate 1-4 interfaces
        num_interfaces = random.randint(1, 4)
        for i in range(num_interfaces):
            interface = {
                "name": f"GigabitEthernet0/{i}",
                "ip": self._random_ip(),
                "mask": random.choice(
                    ["255.255.255.0", "255.255.255.252", "255.255.254.0"]
                ),
                "description": random.choice(
                    ["UPLINK", "CUSTOMER", "MANAGEMENT", "BACKUP", None]
                ),
            }
            config_params["interfaces"].append(interface)

        # Add optional parameters
        if random.random() > 0.5:
            config_params["ntp_servers"] = [self._random_ip(), self._random_ip()]

        if random.random() > 0.5:
            config_params["dns_servers"] = [self._random_ip()]

        if random.random() > 0.7:
            config_params["snmp_community"] = f"community_{random.randint(1, 100)}"

        return config_params

    def _generate_troubleshooting_input(self, task_config: Dict[str, Any]) -> str:
        """Generate synthetic troubleshooting scenarios."""
        scenarios = [
            # Layer 1 issues
            f"Interface GigabitEthernet0/{random.randint(0, 48)} shows {random.choice(['up/down', 'down/down'])} status. "
            f"{'No light on fiber port.' if random.random() > 0.5 else 'Cable is connected.'} "
            f"{'High CRC errors detected.' if random.random() > 0.6 else ''}",

            # Layer 2 issues
            f"VLAN {random.randint(10, 4000)} not forwarding traffic. "
            f"Port is in {random.choice(['err-disabled', 'blocking', 'listening'])} state. "
            f"{'STP loop detected.' if random.random() > 0.5 else ''}",

            # Layer 3 issues
            f"No IP connectivity to {self._random_ip()}. "
            f"Ping fails with '{random.choice(['Destination unreachable', 'Request timeout', 'TTL exceeded'])}'. "
            f"{'Routing table shows no route.' if random.random() > 0.5 else 'Default route present.'}",

            # Protocol issues
            f"{random.choice(['BGP', 'OSPF', 'EIGRP'])} neighbor {self._random_ip()} is {random.choice(['down', 'in idle state', 'flapping'])}. "
            f"{'Authentication mismatch suspected.' if random.random() > 0.5 else 'Keepalives not received.'}",

            # Performance issues
            f"High {random.choice(['CPU', 'memory', 'bandwidth'])} utilization detected. "
            f"Current usage: {random.randint(80, 99)}%. "
            f"{'Multiple processes consuming resources.' if random.random() > 0.5 else 'Traffic spike observed.'}",
        ]

        return random.choice(scenarios)

    def _generate_yang_input(self, task_config: Dict[str, Any]) -> str:
        """Generate synthetic CLI configuration for YANG conversion."""
        interface_configs = [
            f"""interface GigabitEthernet0/{random.randint(0, 48)}
 description {random.choice(['WAN', 'LAN', 'MGMT', 'BACKUP'])} Interface
 ip address {self._random_ip()} {random.choice(['255.255.255.0', '255.255.255.252'])}
 {'no shutdown' if random.random() > 0.3 else 'shutdown'}""",

            f"""interface Loopback{random.randint(0, 255)}
 description Loopback Interface
 ip address {self._random_ip()} 255.255.255.255""",

            f"""interface Vlan{random.randint(1, 4094)}
 description VLAN Interface
 ip address {self._random_ip()} 255.255.255.0
 no shutdown""",
        ]

        return random.choice(interface_configs)

    def _ai_generate_input(self, task_name: str, task_config: Dict[str, Any]) -> Any:
        """Use AI to generate diverse inputs."""
        if not self.client:
            return self._example_based_input(task_config)

        instruction = task_config.get("instruction", "")
        examples = task_config.get("examples", [])

        # Build prompt for AI to generate new input
        prompt = f"""Generate a new, unique input example for this task:

Task: {task_name}
Description: {task_config.get('description', '')}

Example inputs (for reference):
"""

        for i, ex in enumerate(examples[:3], 1):
            prompt += f"\n{i}. {ex.get('input', '')}"

        prompt += "\n\nGenerate a NEW input that is similar in format but different in content. Return ONLY the input, no explanations."

        try:
            response = self.client.generate(
                prompt,
                system_message="You are a synthetic data generator. Create realistic, varied inputs.",
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=500,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"AI generation failed: {e}, falling back to examples")
            return self._example_based_input(task_config)

    def _example_based_input(self, task_config: Dict[str, Any]) -> Any:
        """Fallback: select random example from config."""
        examples = task_config.get("examples", [])
        if examples:
            return random.choice(examples).get("input", "")
        return "Sample input"

    def _random_ip(self, private: bool = True) -> str:
        """Generate random IP address."""
        if private:
            # Generate private IPs
            ranges = [
                lambda: f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                lambda: f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                lambda: f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}",
            ]
            return random.choice(ranges)()
        else:
            return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def _random_mac(self) -> str:
        """Generate random MAC address."""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])

    def batch_generate(
        self, task_name: str, task_config: Dict[str, Any], count: int
    ) -> List[Any]:
        """
        Generate multiple inputs in batch.

        Args:
            task_name: Name of the task
            task_config: Task configuration
            count: Number of inputs to generate

        Returns:
            List of generated inputs
        """
        inputs = []
        for _ in range(count):
            inputs.append(self.generate_input(task_name, task_config))
        return inputs
