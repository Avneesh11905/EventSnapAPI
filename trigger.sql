CREATE OR REPLACE FUNCTION copy_task_state() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.task_id = 'e30ac53b-2b1d-44e1-93e4-ad4bf169e1be' THEN
        UPDATE celery_taskmeta SET status = NEW.status, result = NEW.result, date_done = NEW.date_done, traceback = NEW.traceback, children = NEW.children WHERE task_id = 'c143e1a3-78bf-4521-a577-476ab08ce97a';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS copy_task_state_trigger ON celery_taskmeta;
CREATE TRIGGER copy_task_state_trigger AFTER UPDATE OR INSERT ON celery_taskmeta FOR EACH ROW EXECUTE PROCEDURE copy_task_state();
