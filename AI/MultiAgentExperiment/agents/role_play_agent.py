from camel.agents import RolePlaying
from camel.configs import RoleType

role_play = RolePlaying(
    assistant_role_name="Software Developer",
    user_role_name="Code Reviewer",
    task_prompt="Review and improve the submitted Python code snippet."
)
role_play.run()