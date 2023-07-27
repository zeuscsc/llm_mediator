from .llm import LLM
import json

class Formatter:
    def __init__(self, model:LLM,max_repair_count=3):
        self.model=model
        self.max_repair_count=max_repair_count
        self.repair_count=0
    def json(self, json_data, schema=None):
        try:
            json_data=json.loads(json_data)
        except Exception as e:
            print(e)
            json_data=self.repair_json(json_data,schema,e)
        return json_data
    def repair_json(self, json_data, schema=None,error=None):
        system="""Fix the following JSON string to make it valid.
I don't need any extra description in the JSON only give me the JSON.
"""
        if error is not None:
            system+="""Here is the error message during JSON validation:"""+str(error)
        if schema is not None:
            system+="""You can fix the JSON according to the following schema.  Here is the schema:
    """+json.dumps(schema,ensure_ascii=False)
        assistant="""{"""
        user=json_data
        response=self.model.get_response(system,assistant,user)
        self.repair_count+=1
        try:
            json.loads(response)
            return response
        except Exception as e:
            print(e)
            if self.max_repair_count is not None and self.repair_count>=self.max_repair_count:
                raise e
            return self.repair_json(response, schema,e)
    pass