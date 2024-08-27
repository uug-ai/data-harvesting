import shutil
from os.path import basename as pbasename

import roboflow

from utils.VariableClass import VariableClass


class RoboflowIntegration:
    def __init__(self):
        self._var = VariableClass()
        self.agent, self.ws, self.project = self.__connect__()

    def __connect__(self):
        try:
            # Attempt to initialize Roboflow with the API key
            agent = roboflow.Roboflow(api_key=self._var.ROBOFLOW_API_KEY)

            # Access the workspace
            workspace = agent.workspace(self._var.ROBOFLOW_WORKSPACE)

            # Access the project
            project = workspace.project(self._var.ROBOFLOW_PROJECT)

            return agent, workspace, project

        except Exception as e:
            # Handle any exceptions
            raise ConnectionRefusedError(f'Error during Roboflow login: {e}')

    def upload_dataset(self, src_project_path):
        # Upload data set to an existing project
        self.ws.upload_dataset(
                src_project_path,
                pbasename(self.project.id),
                num_workers=10,
                project_license="MIT",
                project_type="object-detection",
                batch_name=None,
                num_retries=0
        )
        print('Uploaded')

        # Remove local folder when uploaded
        shutil.rmtree(src_project_path)